#!/bin/bash
set -euo pipefail

json_response() {
  local code="$1"
  local message="$2"
  local status="$3"
  local epoch="$4"
  local loss_d="$5"
  local loss_g="$6"
  local loss_adv="$7"
  local loss_g_gan="$8"
  local loss_hinge="$9"

  RESPONSE_CODE="${code}" \
  RESPONSE_MESSAGE="${message}" \
  RESPONSE_STATUS="${status}" \
  RESPONSE_EPOCH="${epoch}" \
  RESPONSE_LOSS_D="${loss_d}" \
  RESPONSE_LOSS_G="${loss_g}" \
  RESPONSE_LOSS_ADV="${loss_adv}" \
  RESPONSE_LOSS_G_GAN="${loss_g_gan}" \
  RESPONSE_LOSS_HINGE="${loss_hinge}" \
  python3 - <<'PY'
import json
import os


def nullable_number(value):
    if value == "" or value.lower() == "null":
        return None

    try:
        if "." in value:
            return float(value)
        return int(value)
    except Exception:
        return None


payload = {
    "code": int(os.environ["RESPONSE_CODE"]),
    "message": os.environ["RESPONSE_MESSAGE"],
    "data": {
        "epoch": nullable_number(os.environ["RESPONSE_EPOCH"]),
        "loss_D": nullable_number(os.environ["RESPONSE_LOSS_D"]),
        "loss_G": nullable_number(os.environ["RESPONSE_LOSS_G"]),
        "loss_adv": nullable_number(os.environ["RESPONSE_LOSS_ADV"]),
        "loss_G_gan": nullable_number(os.environ["RESPONSE_LOSS_G_GAN"]),
        "loss_hinge": nullable_number(os.environ["RESPONSE_LOSS_HINGE"]),
        "status": os.environ["RESPONSE_STATUS"]
    }
}

# print(json.dumps(payload, ensure_ascii=False))
print(json.dumps(payload, ensure_ascii=False, indent=4))
PY
}

fail_json() {
  local message="$1"
  json_response 400 "任务失败: ${message}" "3" "null" "null" "null" "null" "null" "null"
  exit 1
}

parse_poll_file() {
  local poll_file="$1"

  if [ ! -f "${poll_file}" ]; then
    echo "null|null|null|null|null|null"
    return 0
  fi

  python3 - "${poll_file}" <<'PY'
import re
import sys

poll_file = sys.argv[1]

try:
    with open(poll_file, "r", encoding="utf-8") as f:
        text = f.read()
except FileNotFoundError:
    print("null|null|null|null|null|null")
    raise SystemExit(0)

patterns = {
    "epoch": r"Epoch\s+([0-9]+)\s*:",
    "loss_D": r"Loss\s+D:\s*([-+0-9.eE]+)",
    "loss_G": r"Loss\s+G:\s*([-+0-9.eE]+)",
    "loss_adv": r"-Loss\s+Adv:\s*([-+0-9.eE]+)",
    "loss_G_gan": r"-Loss\s+G\s+GAN:\s*([-+0-9.eE]+)",
    "loss_hinge": r"-Loss\s+Hinge:\s*([-+0-9.eE]+)",
}

values = []

for key in ["epoch", "loss_D", "loss_G", "loss_adv", "loss_G_gan", "loss_hinge"]:
    match = re.search(patterns[key], text)
    values.append(match.group(1) if match else "null")

print("|".join(values))
PY
}

if [ "$#" -lt 1 ]; then
  fail_json "missing required argument mission_id"
fi

if [ "$#" -gt 1 ]; then
  fail_json "too many arguments. Usage: bash run_poll.sh <mission_id>"
fi

mission_id="$1"

# mission_id 只允许安全字符，避免路径穿越和命令注入
if ! [[ "${mission_id}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  fail_json "invalid mission_id '${mission_id}'. Allowed characters: A-Z a-z 0-9 _ . -"
fi

APP_DIR="/app"
ADV_SAMPLE_DIR="${APP_DIR}/adv_sample"
ADV_EVAL_DIR="${APP_DIR}/adv_eval"

PID_FILE="/tmp/pid_${mission_id}"
POLL_FILE="${ADV_EVAL_DIR}/poll_${mission_id}.txt"

# 你当前默认模型是 inception_v3，所以按这个文件判断完成。
# 同时兼容你描述里少写 .gz 的 .tar 文件。
TAR_GZ_PATH="${ADV_SAMPLE_DIR}/Attack_generation_inception_v3_${mission_id}.tar.gz"
TAR_PATH="${ADV_SAMPLE_DIR}/Attack_generation_inception_v3_${mission_id}.tar"

IFS='|' read -r epoch loss_d loss_g loss_adv loss_g_gan loss_hinge < <(parse_poll_file "${POLL_FILE}")

if [ -f "${PID_FILE}" ]; then
  status="2"
elif [ -f "${TAR_GZ_PATH}" ] || [ -f "${TAR_PATH}" ]; then
  status="1"
else
  status="3"
fi

json_response 200 "任务成功" "${status}" \
  "${epoch}" \
  "${loss_d}" \
  "${loss_g}" \
  "${loss_adv}" \
  "${loss_g_gan}" \
  "${loss_hinge}"
