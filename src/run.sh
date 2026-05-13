#!/bin/bash
set -euo pipefail

usage() {
  echo "Usage:"
  echo "  bash run.sh <mission_id> [advgan_epochs] [l_inf_bound] [advgan_lr] [timeout_sec] [model_name]"
  echo ""
  echo "Example:"
  echo "  bash run.sh mission001 15 0.05 0.001 3600 inception_v3"
}

json_response() {
  local code="$1"
  local message="$2"
  local status="$3"

  if command -v python3 >/dev/null 2>&1; then
    RESPONSE_CODE="${code}" \
    RESPONSE_MESSAGE="${message}" \
    RESPONSE_STATUS="${status}" \
    python3 - <<'PY'
import json
import os

print(json.dumps({
    "code": int(os.environ["RESPONSE_CODE"]),
    "message": os.environ["RESPONSE_MESSAGE"],
    "data": {
        "status": os.environ["RESPONSE_STATUS"]
    }
}, ensure_ascii=False, indent=4))
PY
  else
    message="${message//\\/\\\\}"
    message="${message//\"/\\\"}"
    cat <<EOF
{
    "code": ${code},
    "message": "${message}",
    "data": {
        "status": "${status}"
    }
}
EOF
  fi
}

fail_json() {
  local message="$1"
  json_response 400 "任务失败: ${message}" "2"
  exit 1
}

success_json() {
  json_response 200 "任务成功" "1"
}

is_positive_int() {
  [[ "$1" =~ ^[0-9]+$ ]] && [ "$1" -gt 0 ]
}

is_positive_number() {
  [[ "$1" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || return 1
  awk -v n="$1" 'BEGIN { exit !(n > 0) }'
}

validate_args() {
  if [ "$#" -lt 1 ]; then
    fail_json "missing required argument mission_id. $(usage | tr '\n' ' ')"
  fi

  if [ "$#" -gt 6 ]; then
    fail_json "too many arguments. $(usage | tr '\n' ' ')"
  fi

  if ! [[ "${mission_id}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    fail_json "invalid mission_id '${mission_id}'. Allowed characters: A-Z a-z 0-9 _ . -"
  fi

  if ! is_positive_int "${advgan_epochs}"; then
    fail_json "advgan_epochs must be a positive integer, got '${advgan_epochs}'"
  fi

  if ! is_positive_number "${l_inf_bound}"; then
    fail_json "l_inf_bound must be a positive number, got '${l_inf_bound}'"
  fi

  if ! is_positive_number "${advgan_lr}"; then
    fail_json "advgan_lr must be a positive number, got '${advgan_lr}'"
  fi

  if ! is_positive_int "${timeout_sec}"; then
    fail_json "timeout_sec must be a positive integer, got '${timeout_sec}'"
  fi

  case "${model_name}" in
    "inception_v3"|"inception"|"resnet50"|"resnet"|"vgg16"|"vgg19")
      ;;
    *)
      fail_json "invalid model_name '${model_name}'. Supported: inception_v3, inception, resnet50, resnet, vgg16, vgg19"
      ;;
  esac
}

require_command() {
  local cmd="$1"
  command -v "${cmd}" >/dev/null 2>&1 || fail_json "required command not found: ${cmd}"
}

validate_runtime() {
  require_command python3
  require_command awk
  require_command flock
  require_command timeout
  require_command unzip
  require_command find
  require_command tar
  require_command tee

  if [ ! -d "${SRC_DIR}" ]; then
    fail_json "source directory not found: ${SRC_DIR}"
  fi

  if [ ! -f "${SRC_DIR}/main_test.py" ]; then
    fail_json "main_test.py not found: ${SRC_DIR}/main_test.py"
  fi

  mkdir -p \
    "${SEED_ROOT}" \
    "${WEIGHT_ROOT}" \
    "${WORK_ROOT}" \
    "${ADV_SAMPLE_DIR}" \
    "${ADV_EVAL_DIR}" \
    || fail_json "failed to create application directories under ${APP_DIR}"
}

wait_for_file() {
  local file_path="$1"
  local max_wait="$2"
  local waited=0

  while [ ! -f "${file_path}" ]; do
    if [ "${waited}" -ge "${max_wait}" ]; then
      echo "Timeout waiting for file: ${file_path}"
      return 1
    fi

    sleep 1
    waited=$((waited + 1))
  done

  return 0
}

worker_main() {
  trap cleanup EXIT

  echo "Mission ${mission_id} accepted."
  echo "Shell PID: $$"
  echo "Run ID: ${RUN_ID}"
  echo "Work directory: ${WORK_DIR}"

  rm -f "${TAR_PATH}"
  rm -f "${EVAL_TXT}"

  echo "Waiting for seed zip: ${SEED_ZIP}"
  wait_for_file "${SEED_ZIP}" 600

  echo "Extracting seed zip..."
  unzip -q "${SEED_ZIP}" -d "${SEED_EXTRACT_DIR}"

  IMAGE_DIR="$(find "${SEED_EXTRACT_DIR}" -type d -name "img" | head -n 1 || true)"
  LABEL_CSV="$(find "${SEED_EXTRACT_DIR}" -type f -name "images.csv" | head -n 1 || true)"

  if [ -z "${IMAGE_DIR}" ] || [ ! -d "${IMAGE_DIR}" ]; then
    echo "Image directory 'img' not found after extracting ${SEED_ZIP}"
    exit 1
  fi

  if [ -z "${LABEL_CSV}" ] || [ ! -f "${LABEL_CSV}" ]; then
    echo "Label file 'images.csv' not found after extracting ${SEED_ZIP}"
    exit 1
  fi

  echo "Image dir: ${IMAGE_DIR}"
  echo "Label CSV: ${LABEL_CSV}"

  TARGET_WEIGHT_PATH=""

  if [ -f "${WEIGHT_ZIP}" ]; then
    echo "Extracting weight zip: ${WEIGHT_ZIP}"
    unzip -q "${WEIGHT_ZIP}" -d "${WEIGHT_EXTRACT_DIR}"

    TARGET_WEIGHT_PATH="$(find "${WEIGHT_EXTRACT_DIR}" -type f \( -name "*.pth" -o -name "*.pt" \) | head -n 1 || true)"

    if [ -z "${TARGET_WEIGHT_PATH}" ]; then
      echo "Weight zip exists but no .pth or .pt file found: ${WEIGHT_ZIP}"
      exit 1
    fi

    echo "Target weight: ${TARGET_WEIGHT_PATH}"
  else
    echo "Weight zip not found: ${WEIGHT_ZIP}"
    echo "Will use torchvision DEFAULT pretrained weight for ${CANONICAL_MODEL_NAME}."
  fi

  echo "Generating mission config: ${CONFIG_PATH}"

  MISSION_ID="${mission_id}" \
  MODEL_NAME="${CANONICAL_MODEL_NAME}" \
  TARGET_WEIGHT_PATH="${TARGET_WEIGHT_PATH}" \
  IMAGE_DIR="${IMAGE_DIR}" \
  LABEL_CSV="${LABEL_CSV}" \
  CHECKPOINT_DIR="${CHECKPOINT_DIR}" \
  LOSS_DIR="${LOSS_DIR}" \
  ADV_IMAGE_DIR="${ADV_IMAGE_DIR}" \
  NPY_DIR="${NPY_DIR}" \
  EVAL_TXT="${EVAL_TXT}" \
  ADVGAN_EPOCHS="${advgan_epochs}" \
  ADVGAN_LR="${advgan_lr}" \
  L_INF_BOUND="${l_inf_bound}" \
  CONFIG_PATH="${CONFIG_PATH}" \
  python3 - <<'PY'
import json
import os

cfg = {
    "mission_id": os.environ["MISSION_ID"],

    "target_dataset": "HighResolution",
    "target_model_name": os.environ["MODEL_NAME"],
    "target_weight_path": os.environ["TARGET_WEIGHT_PATH"],

    "image_dir": os.environ["IMAGE_DIR"],
    "label_csv": os.environ["LABEL_CSV"],

    "checkpoint_dir": os.environ["CHECKPOINT_DIR"],
    "loss_dir": os.environ["LOSS_DIR"],
    "adv_image_dir": os.environ["ADV_IMAGE_DIR"],
    "npy_dir": os.environ["NPY_DIR"],
    "eval_txt": os.environ["EVAL_TXT"],

    "save_npy": False,
    "shuffle": False,
    "num_workers": 0,
    "batch_size": 30,
    "n_labels": 1000,
    "use_cuda": True,

    "target_learning_rate": 0.001,
    "target_model_epochs": 50,

    "AdvGAN_epochs": int(os.environ["ADVGAN_EPOCHS"]),
    "AdvGAN_learning_rate": float(os.environ["ADVGAN_LR"]),
    "maximum_perturbation_allowed": float(os.environ["L_INF_BOUND"]),

    "alpha": 5,
    "beta": 1,
    "gamma": 1,
    "kappa": 0,
    "c": 0.1,
    "D_number_of_steps_per_batch": 1,
    "G_number_of_steps_per_batch": 1,
    "is_relativistic": "True"
}

config_path = os.environ["CONFIG_PATH"]
os.makedirs(os.path.dirname(config_path), exist_ok=True)

with open(config_path, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2, ensure_ascii=False)

print(config_path)
PY

  if [ ! -f "${CONFIG_PATH}" ]; then
    echo "Config file was not generated: ${CONFIG_PATH}"
    exit 1
  fi

  echo "Running mission ${mission_id}..."
  echo "Log file: ${LOG_DIR}/run.log"

  cd "${SRC_DIR}"

  set +e
  timeout --kill-after=30s "${timeout_sec}" \
    python3 -u main_test.py --config "${CONFIG_PATH}" &

  pid=$!
  echo "${pid}" > "${PID_FILE}"

  wait "${pid}"
  exit_code=$?
  set -e

  if [ "${exit_code}" -eq 124 ]; then
    echo "Timeout reached. Mission ${mission_id} stopped."
    exit 124
  elif [ "${exit_code}" -ne 0 ]; then
    echo "Mission ${mission_id} failed with exit code ${exit_code}."
    echo "Check log: ${LOG_DIR}/run.log"
    exit "${exit_code}"
  fi

  if [ ! -d "${ADV_IMAGE_DIR}" ]; then
    echo "Adversarial image directory not found: ${ADV_IMAGE_DIR}"
    exit 1
  fi

  if ! find "${ADV_IMAGE_DIR}" -type f -name "*.png" | grep -q .; then
    echo "No adversarial PNG images generated in ${ADV_IMAGE_DIR}"
    exit 1
  fi

  if [ ! -f "${EVAL_TXT}" ]; then
    echo "Evaluation txt not found: ${EVAL_TXT}"
    exit 1
  fi

  echo "Compressing adversarial images..."

  tar -czf "${TMP_TAR_PATH}" \
    -C "$(dirname "${ADV_IMAGE_DIR}")" \
    "${OUTPUT_DIR_NAME}"

  mv -f "${TMP_TAR_PATH}" "${TAR_PATH}"

  echo "download adv_sample ${TAR_NAME} in ${ADV_SAMPLE_DIR}"
  echo "download adv_eval ${mission_id}.txt in ${ADV_EVAL_DIR}"
  echo "Mission ${mission_id} completed successfully."
}

cleanup() {
  status=$?

  rm -f "${PID_FILE}"
  rm -f "${TMP_TAR_PATH}"

  # if [ "${status}" -eq 0 ]; then
  #   rm -rf "${WORK_DIR}"
  # else
  #   echo "Mission failed. Work directory kept for debugging: ${WORK_DIR}"
  # fi

  exit "${status}"
}

mission_id="${1:-}"
advgan_epochs="${2:-15}"
l_inf_bound="${3:-0.05}"
advgan_lr="${4:-0.001}"
timeout_sec="${5:-3600}"
model_name="${6:-inception_v3}"

APP_DIR="/app"
SRC_DIR="${APP_DIR}/src"

SEED_ROOT="${APP_DIR}/seed"
WEIGHT_ROOT="${APP_DIR}/weight"
WORK_ROOT="${APP_DIR}/work"
ADV_SAMPLE_DIR="${APP_DIR}/adv_sample"
ADV_EVAL_DIR="${APP_DIR}/adv_eval"

validate_args "$@"

CANONICAL_MODEL_NAME="${model_name}"
if [ "${CANONICAL_MODEL_NAME}" = "inception" ]; then
  CANONICAL_MODEL_NAME="inception_v3"
elif [ "${CANONICAL_MODEL_NAME}" = "resnet" ]; then
  CANONICAL_MODEL_NAME="resnet50"
fi

OUTPUT_BASENAME="Attack_generation_${CANONICAL_MODEL_NAME}_${mission_id}"
OUTPUT_DIR_NAME="${OUTPUT_BASENAME}"

RUN_ID="${mission_id}_$(date +%Y%m%d%H%M%S)_$$"
WORK_DIR="${WORK_ROOT}/${RUN_ID}"

SEED_ZIP="${SEED_ROOT}/${mission_id}.zip"
WEIGHT_ZIP="${WEIGHT_ROOT}/${mission_id}.zip"

SEED_EXTRACT_DIR="${WORK_DIR}/seed"
WEIGHT_EXTRACT_DIR="${WORK_DIR}/weight"
CONFIG_DIR="${WORK_DIR}/config"
CHECKPOINT_DIR="${WORK_DIR}/checkpoints/AdvGAN"
LOSS_DIR="${WORK_DIR}/results/losses"
ADV_IMAGE_DIR="${WORK_DIR}/results/examples/${OUTPUT_DIR_NAME}"
NPY_DIR="${WORK_DIR}/npy"
LOG_DIR="${WORK_DIR}/logs"

CONFIG_PATH="${CONFIG_DIR}/hyperparams_${mission_id}.json"
EVAL_TXT="${ADV_EVAL_DIR}/${mission_id}.txt"

TAR_NAME="${OUTPUT_BASENAME}.tar.gz"
TAR_PATH="${ADV_SAMPLE_DIR}/${TAR_NAME}"
TMP_TAR_PATH="${TAR_PATH}.tmp.$$"

LOCK_FILE="/tmp/gan_attack_${mission_id}.lock"
PID_FILE="/tmp/pid_${mission_id}"

validate_runtime

mkdir -p \
  "${SEED_EXTRACT_DIR}" \
  "${WEIGHT_EXTRACT_DIR}" \
  "${CONFIG_DIR}" \
  "${CHECKPOINT_DIR}" \
  "${LOSS_DIR}" \
  "${ADV_IMAGE_DIR}" \
  "${NPY_DIR}" \
  "${LOG_DIR}" \
  || fail_json "failed to create work directories: ${WORK_DIR}"

exec 9>"${LOCK_FILE}"
flock -n 9 || fail_json "mission_id ${mission_id} is already running"

: > "${LOG_DIR}/run.log"

success_json

exec > >(tee -a "${LOG_DIR}/run.log") 2>&1

worker_main