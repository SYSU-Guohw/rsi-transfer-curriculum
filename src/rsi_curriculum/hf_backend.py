"""Qwen3 + PEFT + TRL backend for real curriculum branch training."""

import gc
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .execution import TransitionCache
from .trajectory import execute_completion, make_grpo_reward


@dataclass(frozen=True)
class HFBackendConfig:
    model_name: str = "Qwen/Qwen3-8B"
    model_revision: str = "b968826d9c46dd6066d109eabc6255188de91218"
    dtype: str = "bfloat16"
    load_in_4bit: bool = False
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    learning_rate: float = 1e-5
    warmup_steps: int = 20
    branch_steps: int = 8
    generations_per_prompt: int = 4
    max_prompt_length: int = 1536
    max_completion_length: int = 768
    temperature: float = 0.8
    beta: float = 0.01
    gradient_accumulation_steps: int = 1


def _imports():
    try:
        import torch
        from datasets import Dataset
        from peft import (
            LoraConfig,
            PeftModel,
            get_peft_model,
            prepare_model_for_kbit_training,
        )
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            set_seed,
        )
        from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError(
            "The hf backend requires the train extra: pip install -e '.[train]'"
        ) from exc
    return {
        "torch": torch,
        "Dataset": Dataset,
        "LoraConfig": LoraConfig,
        "PeftModel": PeftModel,
        "get_peft_model": get_peft_model,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "set_seed": set_seed,
        "GRPOConfig": GRPOConfig,
        "GRPOTrainer": GRPOTrainer,
        "SFTConfig": SFTConfig,
        "SFTTrainer": SFTTrainer,
    }


def _torch_load(torch, path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class HFBackend:
    name = "hf"

    def __init__(self, registry, config, seed=0):
        if not isinstance(config, HFBackendConfig):
            raise TypeError("config must be HFBackendConfig")
        self.registry = registry
        self.config = config
        self.seed = seed
        self.transition_cache = TransitionCache()

    def _load_model(self, adapter_checkpoint=None):
        libs = _imports()
        torch = libs["torch"]
        if not torch.cuda.is_available():
            raise RuntimeError("The hf backend requires a CUDA GPU")
        dtype = getattr(torch, self.config.dtype)
        kwargs = {
            "revision": self.config.model_revision,
            "dtype": dtype,
            "trust_remote_code": False,
        }
        if self.config.load_in_4bit:
            kwargs["quantization_config"] = libs["BitsAndBytesConfig"](
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            kwargs["device_map"] = {"": int(torch.cuda.current_device())}
        model = libs["AutoModelForCausalLM"].from_pretrained(
            self.config.model_name, **kwargs
        )
        if self.config.load_in_4bit:
            model = libs["prepare_model_for_kbit_training"](
                model, use_gradient_checkpointing=True
            )
        if adapter_checkpoint is None:
            peft_config = libs["LoraConfig"](
                task_type="CAUSAL_LM",
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=(
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ),
            )
            model = libs["get_peft_model"](model, peft_config)
        else:
            model = libs["PeftModel"].from_pretrained(
                model, str(Path(adapter_checkpoint) / "adapter"), is_trainable=True
            )
        tokenizer = libs["AutoTokenizer"].from_pretrained(
            self.config.model_name,
            revision=self.config.model_revision,
            trust_remote_code=False,
            padding_side="left",
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        return libs, model, tokenizer

    def _save(self, destination, model, optimizer, metadata):
        libs = _imports()
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(destination / "adapter", safe_serialization=True)
        libs["torch"].save(optimizer.state_dict(), destination / "optimizer.pt")
        (destination / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _empty_cache(libs):
        gc.collect()
        libs["torch"].cuda.empty_cache()

    def initialize(self, destination, warmup_tasks):
        tasks = tuple(warmup_tasks)
        libs, model, tokenizer = self._load_model()
        libs["set_seed"](self.seed)
        dataset = libs["Dataset"].from_list(
            [
                {
                    "prompt": task.messages,
                    "completion": [
                        {"role": "assistant", "content": task.oracle_completion()}
                    ],
                }
                for task in tasks
            ]
        )
        steps = max(self.config.warmup_steps, 1)
        args = libs["SFTConfig"](
            output_dir=str(Path(destination) / "trainer"),
            max_steps=steps,
            learning_rate=self.config.learning_rate,
            lr_scheduler_type="constant",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            max_length=self.config.max_prompt_length
            + self.config.max_completion_length,
            completion_only_loss=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            bf16=self.config.dtype == "bfloat16",
            fp16=self.config.dtype == "float16",
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            seed=self.seed,
            data_seed=self.seed,
        )
        trainer = libs["SFTTrainer"](
            model=model,
            args=args,
            train_dataset=dataset,
            processing_class=tokenizer,
        )
        result = trainer.train()
        self._save(
            destination,
            trainer.model,
            trainer.optimizer,
            {
                "kind": "initial-sft",
                "backend": "hf",
                "config": asdict(self.config),
                "examples": len(tasks),
                "steps": steps,
            },
        )
        metrics = dict(result.metrics)
        del trainer, model, tokenizer, dataset
        self._empty_cache(libs)
        return metrics

    def train_branch(self, source, destination, tasks, branch_seed):
        tasks = tuple(tasks)
        libs, model, tokenizer = self._load_model(source)
        libs["set_seed"](branch_seed)
        dataset = libs["Dataset"].from_list(
            [{"prompt": task.messages, "task_json": task.to_json()} for task in tasks]
        )
        generations = self.config.generations_per_prompt
        args = libs["GRPOConfig"](
            output_dir=str(Path(destination) / "trainer"),
            max_steps=self.config.branch_steps,
            learning_rate=self.config.learning_rate,
            lr_scheduler_type="constant",
            per_device_train_batch_size=generations,
            generation_batch_size=generations,
            num_generations=generations,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            max_completion_length=self.config.max_completion_length,
            temperature=self.config.temperature,
            top_p=1.0,
            top_k=0,
            beta=self.config.beta,
            loss_type="dapo",
            scale_rewards="group",
            mask_truncated_completions=True,
            chat_template_kwargs={"enable_thinking": False},
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            bf16=self.config.dtype == "bfloat16",
            fp16=self.config.dtype == "float16",
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            seed=branch_seed,
            data_seed=branch_seed,
        )
        trainer = libs["GRPOTrainer"](
            model=model,
            reward_funcs=make_grpo_reward(
                self.registry,
                metric="reward",
                transition_cache=self.transition_cache,
            ),
            args=args,
            train_dataset=dataset,
            processing_class=tokenizer,
        )
        trainer.create_optimizer_and_scheduler(self.config.branch_steps)
        optimizer_path = Path(source) / "optimizer.pt"
        if optimizer_path.is_file():
            trainer.optimizer.load_state_dict(
                _torch_load(libs["torch"], optimizer_path)
            )
        result = trainer.train()
        self._save(
            destination,
            trainer.model,
            trainer.optimizer,
            {
                "kind": "grpo-branch",
                "backend": "hf",
                "config": asdict(self.config),
                "source": str(source),
                "task_ids": [task.task_id for task in tasks],
                "seed": branch_seed,
            },
        )
        metrics = dict(result.metrics)
        del trainer, model, tokenizer, dataset
        self._empty_cache(libs)
        return metrics

    def evaluate(self, checkpoint, tasks, metric, evaluation_seed):
        if metric not in ("reward", "success"):
            raise ValueError("Unknown evaluation metric")
        libs, model, tokenizer = self._load_model(checkpoint)
        torch = libs["torch"]
        model.eval()
        if not self.config.load_in_4bit:
            model.to(torch.device("cuda", torch.cuda.current_device()))
        results = {}
        details = {}
        with torch.inference_mode():
            for task in tasks:
                digest = hashlib.sha256(
                    f"{evaluation_seed}:{task.task_id}".encode()
                ).digest()
                libs["set_seed"](int.from_bytes(digest[:4], "big"))
                input_ids = tokenizer.apply_chat_template(
                    task.messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                    return_tensors="pt",
                ).to(model.device)
                output = model.generate(
                    input_ids=input_ids,
                    do_sample=True,
                    temperature=self.config.temperature,
                    top_p=1.0,
                    top_k=0,
                    max_new_tokens=self.config.max_completion_length,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                completion = tokenizer.decode(
                    output[0, input_ids.shape[1] :], skip_special_tokens=True
                )
                score = execute_completion(
                    task,
                    completion,
                    self.registry,
                    transition_cache=self.transition_cache,
                )
                results[task.task_id] = getattr(score, metric)
                details[task.task_id] = {
                    **score.to_dict(),
                    "completion": completion,
                }
        del output, input_ids, model, tokenizer
        self._empty_cache(libs)
        return results, details
