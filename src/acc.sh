#!/bin/bash
set -euo pipefail

usage() {
  echo "Usage:"
  echo "  bash acc.sh <mission_id> <model_name> <seed> <weight>"
  echo ""
  echo "Example:"
  echo "  bash acc.sh 20260502 inception_v3 20260502 None"
  echo "  bash acc.sh 20260502 inception_v3 20260502 20260502"
  echo ""
  echo "Notes:"
  echo "  model_name currently only supports inception_v3/inception."
  echo "  seed is kept for API compatibility. Current calc_seed_acc.py reads /app/seed/<mission_id>.zip."
  echo "  weight=None uses default /app/weight/inception_v3_google-0cc3c7bd(.pth/.pt)."
  echo "  weight!=None uses /app/weight/<mission_id>.zip in calc_seed_acc.py."
}

json_response() {
  local code="$1"
  local message="$2"
  local status="$3"

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
}

fail_json() {
  local message="$1"
  json_response 400 "任务失败: ${message}" "2"
  exit 1
}

success_json() {
  json_response 200 "任务成功" "1"
}

is_none_arg() {
  local value="${1:-}"
  value="$(printf '%s' "${value}" | tr '[:upper:]' '[:lower:]')"
  [ "${value}" = "none" ] || [ -z "${value}" ]
}

validate_safe_token() {
  local name="$1"
  local value="$2"

  if ! [[ "${value}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    fail_json "invalid ${name} '${value}'. Allowed characters: A-Z a-z 0-9 _ . -"
  fi
}

require_command() {
  local cmd="$1"
  command -v "${cmd}" >/dev/null 2>&1 || fail_json "required command not found: ${cmd}"
}

validate_args() {
  if [ "$#" -lt 4 ]; then
    fail_json "missing required arguments. $(usage | tr '\n' ' ')"
  fi

  if [ "$#" -gt 4 ]; then
    fail_json "too many arguments. $(usage | tr '\n' ' ')"
  fi

  validate_safe_token "mission_id" "${mission_id}"

  case "${model_name}" in
    "inception_v3"|"inception")
      ;;
    *)
      fail_json "invalid model_name '${model_name}'. ACC script currently supports only inception_v3/inception."
      ;;
  esac

  if ! is_none_arg "${seed_arg}"; then
    validate_safe_token "seed" "${seed_arg}"
  fi

  if ! is_none_arg "${weight_arg}"; then
    validate_safe_token "weight" "${weight_arg}"
  fi
}

cleanup_by_mission() {
  echo "Cleaning previous ACC files for mission_id=${mission_id}..."

  rm -f "${PID_FILE}"
  rm -f "${EVAL_TXT}"
  rm -f "${ACC_DETAIL_TXT}"
  rm -f "${LOG_FILE}"

  rm -rf "${ACC_WORK_GLOB_PREFIX}"_* 2>/dev/null || true
}

cleanup_on_exit() {
  status=$?
  rm -f "${PID_FILE}"

  if [ "${status}" -eq 0 ]; then
    echo "ACC mission ${mission_id} completed."
  else
    echo "ACC mission ${mission_id} failed with exit code ${status}."
    echo "Check log: ${LOG_FILE}"
  fi

  exit "${status}"
}

mission_id="${1:-}"
model_name="${2:-}"
seed_arg="${3:-}"
weight_arg="${4:-}"

APP_DIR="/app"
SRC_DIR="${APP_DIR}/src"
SEED_ROOT="${APP_DIR}/seed"
WEIGHT_ROOT="${APP_DIR}/weight"
ADV_EVAL_DIR="${APP_DIR}/adv_eval"
ACC_RESULT_DIR="${APP_DIR}/ACC_result"
WORK_ROOT="${APP_DIR}/work"

PY_SCRIPT="${SRC_DIR}/calc_seed_acc.py"

EVAL_TXT="${ADV_EVAL_DIR}/eval_${mission_id}.txt"
ACC_DETAIL_TXT="${ACC_RESULT_DIR}/ACC_${mission_id}.txt"
LOG_FILE="${ACC_RESULT_DIR}/ACC_${mission_id}.log"

PID_FILE="/tmp/acc_pid_${mission_id}"
LOCK_FILE="/tmp/acc_${mission_id}.lock"

ACC_WORK_GLOB_PREFIX="${WORK_ROOT}/ACC_${mission_id}"

validate_args "$@"

require_command python3
require_command tee
require_command flock

if [ ! -f "${PY_SCRIPT}" ]; then
  fail_json "python script not found: ${PY_SCRIPT}"
fi

mkdir -p "${ADV_EVAL_DIR}" "${ACC_RESULT_DIR}" "${WORK_ROOT}" \
  || fail_json "failed to create output directories"

# 同一个 mission_id 的 ACC 任务不允许并发
exec 9>"${LOCK_FILE}"
flock -n 9 || fail_json "ACC mission_id ${mission_id} is already running"

# 参数检查通过后，先返回 JSON
success_json

# 后续内容继续正常输出到终端，同时写日志
: > "${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1

trap cleanup_on_exit EXIT

cleanup_by_mission

# 前台任务的 PID 文件：记录 acc.sh 当前 shell PID。
# 任务运行期间该文件存在；脚本退出时 cleanup_on_exit 会删除。
echo "$$" > "${PID_FILE}"

echo "ACC mission ${mission_id} accepted."
echo "Shell PID: $$"
echo "PID file: ${PID_FILE}"
echo "Model name: ${model_name}"
echo "Seed argument: ${seed_arg}"
echo "Weight argument: ${weight_arg}"
echo "Python script: ${PY_SCRIPT}"
echo "Log file: ${LOG_FILE}"

if ! is_none_arg "${seed_arg}"; then
  echo "Expected seed zip: ${SEED_ROOT}/${mission_id}.zip"
else
  echo "Seed argument is None."
  echo "Note: current calc_seed_acc.py must support default seed if you want seed=None."
fi

if is_none_arg "${weight_arg}"; then
  echo "Using default weight in calc_seed_acc.py."
else
  echo "Expected weight zip: ${WEIGHT_ROOT}/${mission_id}.zip"
fi

echo "Running ACC calculation..."

# 不后台跑：这里直接前台执行，终端会继续输出 calc_seed_acc.py 的日志。
python3 -u "${PY_SCRIPT}" "${mission_id}" "${weight_arg}"

echo "ACC result txt: ${EVAL_TXT}"
echo "ACC detail txt: ${ACC_DETAIL_TXT}"
echo "ACC mission ${mission_id} finished successfully."
