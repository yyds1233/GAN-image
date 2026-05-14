#!/bin/bash
set -euo pipefail

usage() {
  echo "Usage:"
  echo "  bash acc_poll.sh <mission_id>"
  echo ""
  echo "Example:"
  echo "  bash acc_poll.sh 20260502"
}

json_response() {
  local code="$1"
  local message="$2"
  local status="$3"
  local acc="$4"

  RESPONSE_CODE="${code}" \
  RESPONSE_MESSAGE="${message}" \
  RESPONSE_STATUS="${status}" \
  RESPONSE_ACC="${acc}" \
  python3 - <<'PY'
import json
import os

acc_raw = os.environ["RESPONSE_ACC"]

if acc_raw == "" or acc_raw.lower() == "null":
    acc_value = None
else:
    acc_value = acc_raw

print(json.dumps({
    "code": int(os.environ["RESPONSE_CODE"]),
    "message": os.environ["RESPONSE_MESSAGE"],
    "data": {
        "status": os.environ["RESPONSE_STATUS"],
        "ACC": acc_value
    }
}, ensure_ascii=False, indent=4))
PY
}

fail_json() {
  local message="$1"
  json_response 400 "任务失败: ${message}" "3" "null"
  exit 1
}

validate_safe_token() {
  local name="$1"
  local value="$2"

  if ! [[ "${value}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    fail_json "invalid ${name} '${value}'. Allowed characters: A-Z a-z 0-9 _ . -"
  fi
}

read_acc_value() {
  local eval_file="$1"

  if [ ! -f "${eval_file}" ]; then
    echo "null"
    return 0
  fi

  python3 - "${eval_file}" <<'PY'
import re
import sys

path = sys.argv[1]

try:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
except FileNotFoundError:
    print("null")
    raise SystemExit(0)

# 支持:
# ACC: 76.50
# ACC:76.50
# 也兼容直接只有数字的情况
m = re.search(r"ACC\s*:\s*([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)

if m:
    print(m.group(1))
else:
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", text)
    print(m.group(1) if m else "null")
PY
}

if [ "$#" -lt 1 ]; then
  fail_json "missing required argument mission_id. $(usage | tr '\n' ' ')"
fi

if [ "$#" -gt 1 ]; then
  fail_json "too many arguments. $(usage | tr '\n' ' ')"
fi

mission_id="$1"
validate_safe_token "mission_id" "${mission_id}"

APP_DIR="/app"
ADV_EVAL_DIR="${APP_DIR}/adv_eval"
ACC_RESULT_DIR="${APP_DIR}/ACC_result"

PID_FILE="/tmp/acc_pid_${mission_id}"
EVAL_TXT="${ADV_EVAL_DIR}/eval_${mission_id}.txt"
ACC_DETAIL_TXT="${ACC_RESULT_DIR}/ACC_${mission_id}.txt"

acc_value="$(read_acc_value "${EVAL_TXT}")"

if [ -f "${PID_FILE}" ]; then
  status="2"
elif [ -f "${ACC_DETAIL_TXT}" ]; then
  status="1"
else
  status="3"
fi

json_response 200 "任务成功" "${status}" "${acc_value}"
