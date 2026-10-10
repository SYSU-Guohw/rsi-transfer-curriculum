#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Submit the Qwen3 pilot to a ByteDance Merlin/Arnold queue.

Required environment variables:
  MERLIN_GROUP_ID       Numeric resource group ID
  MERLIN_CLUSTER_ID     Numeric cluster ID
  MERLIN_QUEUE_NAME     Arnold queue name
  MERLIN_IMAGE_URL      CUDA/PyTorch training image

Optional environment variables:
  MERLIN_SITE           cn (default), i18n-tt, or i18n-bd
  MERLIN_VREGION        Optional Merlin vregion, for example seed
  MERLIN_NAMESPACE      Product namespace; omitted means personal default
  MERLIN_GPU_TYPE       A100_SXM_80GB (default)
  MERLIN_CPU            32 (default)
  MERLIN_MEMORY_MB      131072 (default)
  MERLIN_BYTENAS_NAME   Persistent ByteNAS volume name
  RSI_OUTPUT_DIR        Actual persistent mount path when ByteNAS is used, or
                        /opt/tiger/rsi-transfer-curriculum-output by default
  RSI_REPO_URL          Public Git repository URL
  RSI_GIT_REF           main (default); prefer a full commit SHA
  RSI_CONFIG            configs/qwen3_8b_pilot.json (default)
  RSI_JOB_NAME          rsi-transfer-curriculum-pilot (default)

Options:
  --dry-run             Ask Merlin to validate/preview without submission
  --print-only          Print generated JSON inputs without calling Merlin
  -h, --help            Show this help
EOF
}

dry_run=0
print_only=0
while (($#)); do
  case "$1" in
    --dry-run) dry_run=1 ;;
    --print-only) print_only=1 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

required=(
  MERLIN_GROUP_ID
  MERLIN_CLUSTER_ID
  MERLIN_QUEUE_NAME
  MERLIN_IMAGE_URL
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    printf 'Missing required environment variable: %s\n' "$name" >&2
    exit 2
  fi
done

MERLIN_SITE="${MERLIN_SITE:-cn}"
MERLIN_VREGION="${MERLIN_VREGION:-}"
MERLIN_NAMESPACE="${MERLIN_NAMESPACE:-}"
MERLIN_GPU_TYPE="${MERLIN_GPU_TYPE:-A100_SXM_80GB}"
MERLIN_CPU="${MERLIN_CPU:-32}"
MERLIN_MEMORY_MB="${MERLIN_MEMORY_MB:-131072}"
MERLIN_BYTENAS_NAME="${MERLIN_BYTENAS_NAME:-}"
RSI_REPO_URL="${RSI_REPO_URL:-https://github.com/SYSU-Guohw/rsi-transfer-curriculum.git}"
RSI_GIT_REF="${RSI_GIT_REF:-main}"
RSI_CONFIG="${RSI_CONFIG:-configs/qwen3_8b_pilot.json}"
RSI_JOB_NAME="${RSI_JOB_NAME:-rsi-transfer-curriculum-pilot}"
if [[ -n "$MERLIN_BYTENAS_NAME" ]]; then
  if [[ -z "${RSI_OUTPUT_DIR:-}" ]]; then
    printf '%s\n' \
      'RSI_OUTPUT_DIR must be the actual mounted path when ByteNAS is enabled.' >&2
    exit 2
  fi
else
  RSI_OUTPUT_DIR="${RSI_OUTPUT_DIR:-/opt/tiger/rsi-transfer-curriculum-output}"
  printf '%s\n' \
    'Warning: MERLIN_BYTENAS_NAME is unset; checkpoints may be ephemeral.' >&2
fi

export MERLIN_GROUP_ID MERLIN_CLUSTER_ID MERLIN_QUEUE_NAME MERLIN_GPU_TYPE
export MERLIN_CPU MERLIN_MEMORY_MB MERLIN_BYTENAS_NAME
export RSI_REPO_URL RSI_GIT_REF RSI_CONFIG RSI_OUTPUT_DIR

RESOURCE_CONFIG="$(python3 - <<'PY'
import json
import os

arnold = {
    "group_ids": [int(os.environ["MERLIN_GROUP_ID"])],
    "cluster_id": int(os.environ["MERLIN_CLUSTER_ID"]),
    "roles": [{
        "name": "worker",
        "num": 1,
        "gpu": 1,
        "gpuv": os.environ["MERLIN_GPU_TYPE"],
        "queue_name": os.environ["MERLIN_QUEUE_NAME"],
        "cpu": int(os.environ["MERLIN_CPU"]),
        "memory": int(os.environ["MERLIN_MEMORY_MB"]),
        "ports": 8,
    }],
    "retry_times": 0,
    "use_tensorboard": True,
}
volume = os.environ.get("MERLIN_BYTENAS_NAME")
if volume:
    arnold["bytenas_volumes"] = [{
        "name": volume,
        "access_mode": "RW",
        "roles": ["worker"],
    }]
print(json.dumps({"backend": "ARNOLD", "arnold_config": arnold}, separators=(",", ":")))
PY
)"

JOB_ENV="$(python3 - <<'PY'
import json
import os

keys = ("RSI_REPO_URL", "RSI_GIT_REF", "RSI_CONFIG", "RSI_OUTPUT_DIR")
value = {key: os.environ[key] for key in keys}
value.update({
    "PYTHONUNBUFFERED": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "HF_HOME": os.path.join(os.environ["RSI_OUTPUT_DIR"], "hf-cache"),
})
print(json.dumps(value, separators=(",", ":")))
PY
)"

read -r -d '' ENTRYPOINT <<'REMOTE' || true
set -euo pipefail
WORK_DIR="$(mktemp -d /opt/tiger/rsi-transfer-curriculum.XXXXXX)"
trap 'rm -rf "$WORK_DIR"' EXIT
git clone "$RSI_REPO_URL" "$WORK_DIR"
cd "$WORK_DIR"
git checkout --detach "$RSI_GIT_REF"
python3 -c 'import sys; assert sys.version_info >= (3, 11), "training requires Python 3.11+"'
python3 -m pip install --upgrade pip
python3 -m pip install -e '.[train]'
export PYTHONPATH="$WORK_DIR/src"
python3 -m rsi_curriculum.vendor bootstrap
python3 -m unittest discover -s tests -v
python3 -m rsi_curriculum.training \
  --config "$RSI_CONFIG" \
  --output "$RSI_OUTPUT_DIR"
REMOTE

if ((print_only)); then
  python3 - "$RESOURCE_CONFIG" "$JOB_ENV" "$ENTRYPOINT" <<'PY'
import json
import sys

print(json.dumps({
    "resource_config": json.loads(sys.argv[1]),
    "env": json.loads(sys.argv[2]),
    "entrypoint_full_script": sys.argv[3],
}, indent=2, sort_keys=True))
PY
  exit 0
fi

if command -v bytedcli >/dev/null 2>&1; then
  cli=(bytedcli)
elif command -v npx >/dev/null 2>&1; then
  cli=(env NPM_CONFIG_REGISTRY=http://bnpm.byted.org npx -y @bytedance-dev/bytedcli@latest)
else
  printf '%s\n' 'bytedcli/npx not found; install bytedcli before submission.' >&2
  exit 127
fi

global=(--site "$MERLIN_SITE" --json)
if [[ -n "$MERLIN_VREGION" ]]; then
  global+=(--vregion "$MERLIN_VREGION")
fi
command=(
  "${cli[@]}"
  "${global[@]}"
  merlin job create-run
  --launch-mode from_scratch
  --caption "$RSI_JOB_NAME"
  --comment "Transfer-credit curriculum Qwen3 pilot"
  --job-type Train
  --python-version 3.11
  --image-url "$MERLIN_IMAGE_URL"
  --entrypoint-full-script "$ENTRYPOINT"
  --resource-config "$RESOURCE_CONFIG"
  --env "$JOB_ENV"
  --notify-settings '{"disable_all":false,"rules":[{"channel":"APP","scenes":["JOB_FAILED","JOB_DONE","JOB_QUEUE_TIMEOUT"]}]}'
  --tags '["rsi","transfer-curriculum","pilot"]'
)
if [[ -n "$MERLIN_NAMESPACE" ]]; then
  command+=(--namespace "$MERLIN_NAMESPACE")
fi
if ((dry_run)); then
  command+=(--dry-run)
fi
"${command[@]}"
