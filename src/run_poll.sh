#!/bin/bash

# =========================================
# GAN Image AdvGAN poll 脚本
#
# 用法：
#   ./run_poll.sh '{"mission_id":"2059839135698386944"}'
#
# 状态定义：
#   status: 1 = 正在执行中
#   status: 2 = 已结束 / 查询成功
#   status: 3 = 参数错误或任务失败
#
# 当前 run.sh 对应路径：
#   PID:          /tmp/pid_${mission_id}
#   runner PID:   /tmp/gan_runner_${mission_id}.pid
#   poll file:    /app/adv_eval/poll_${mission_id}.txt
#   eval file:    /app/adv_eval/${mission_id}.txt
#   result zip:   /app/adv_sample/${mission_id}.zip
#   work dir:     /app/work/${mission_id}_*
#
# 返回字段：
#   Datanum       当前已生成/最终打包的对抗样本数量
#   epoch/loss_*  AdvGAN 训练过程中的最新 loss 信息，来自 poll_${mission_id}.txt
# =========================================

# ===============================
# JSON 输出函数
# ===============================
json_param_error() {
    echo "{
    \"code\": 400,
    \"message\": \"查询失败\",
    \"data\": {
        \"msg\": \"参数输入错误\",
        \"status\": \"3\"
    }
}"
}

json_success_done() {
    local datanum="$1"
    local epoch="$2"
    local loss_d="$3"
    local loss_g="$4"
    local loss_adv="$5"
    local loss_g_gan="$6"
    local loss_hinge="$7"

    RESPONSE_DATANUM="${datanum}" \
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
    if value is None:
        return None
    value = str(value)
    if value == "" or value.lower() == "null":
        return None
    try:
        if "." in value or "e" in value.lower():
            return float(value)
        return int(value)
    except Exception:
        return None

payload = {
    "code": 200,
    "message": "查询成功",
    "data": {
        "Datanum": os.environ["RESPONSE_DATANUM"],
        "epoch": nullable_number(os.environ["RESPONSE_EPOCH"]),
        "loss_D": nullable_number(os.environ["RESPONSE_LOSS_D"]),
        "loss_G": nullable_number(os.environ["RESPONSE_LOSS_G"]),
        "loss_adv": nullable_number(os.environ["RESPONSE_LOSS_ADV"]),
        "loss_G_gan": nullable_number(os.environ["RESPONSE_LOSS_G_GAN"]),
        "loss_hinge": nullable_number(os.environ["RESPONSE_LOSS_HINGE"]),
        "status": "2"
    }
}

print(json.dumps(payload, ensure_ascii=False, indent=4))
PY
}

json_running() {
    local datanum="$1"
    local epoch="$2"
    local loss_d="$3"
    local loss_g="$4"
    local loss_adv="$5"
    local loss_g_gan="$6"
    local loss_hinge="$7"

    RESPONSE_DATANUM="${datanum}" \
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
    if value is None:
        return None
    value = str(value)
    if value == "" or value.lower() == "null":
        return None
    try:
        if "." in value or "e" in value.lower():
            return float(value)
        return int(value)
    except Exception:
        return None

payload = {
    "code": 200,
    "message": "任务正在执行中",
    "data": {
        "Datanum": os.environ["RESPONSE_DATANUM"],
        "epoch": nullable_number(os.environ["RESPONSE_EPOCH"]),
        "loss_D": nullable_number(os.environ["RESPONSE_LOSS_D"]),
        "loss_G": nullable_number(os.environ["RESPONSE_LOSS_G"]),
        "loss_adv": nullable_number(os.environ["RESPONSE_LOSS_ADV"]),
        "loss_G_gan": nullable_number(os.environ["RESPONSE_LOSS_G_GAN"]),
        "loss_hinge": nullable_number(os.environ["RESPONSE_LOSS_HINGE"]),
        "status": "1"
    }
}

print(json.dumps(payload, ensure_ascii=False, indent=4))
PY
}

json_task_failed() {
    local msg="${1:-任务失败}"
    echo "{
    \"code\": 200,
    \"message\": \"任务失败\",
    \"data\": {
        \"Datanum\": \"null\",
        \"msg\": \"${msg}\",
        \"status\": \"3\"
    }
}"
}

json_not_exist() {
    echo "{
    \"code\": 1002,
    \"message\": \"任务不存在。\",
    \"data\": {
    }
}"
}

# ===============================
# 参数解析
# ===============================
json_input="${1:-}"

if [ -z "$json_input" ]; then
    json_param_error
    exit 1
fi

mission_id=$(JSON_INPUT="$json_input" python3 - <<'PY' 2>/dev/null
import json
import os

try:
    data = json.loads(os.environ.get("JSON_INPUT", ""))
    print(data.get("mission_id", "None"))
except Exception:
    print("None")
PY
)

# ===============================
# 参数检查
# ===============================
if [ -z "$mission_id" ] || [ "$mission_id" = "None" ]; then
    json_param_error
    exit 1
fi

# 和 run.sh 保持一致：允许字母、数字、下划线、点、中划线
if ! [[ "$mission_id" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    json_param_error
    exit 1
fi

# ===============================
# 路径定义
# ===============================
APP_DIR="/app"
ADV_SAMPLE_DIR="${APP_DIR}/adv_sample"
ADV_EVAL_DIR="${APP_DIR}/adv_eval"
WORK_ROOT="${APP_DIR}/work"
LOG_DIR="${APP_DIR}/run_logs"

PID_FILE="/tmp/pid_${mission_id}"
RUNNER_PID_FILE="/tmp/gan_runner_${mission_id}.pid"

POLL_FILE="${ADV_EVAL_DIR}/poll_${mission_id}.txt"
EVAL_RESULT_FILE="${ADV_EVAL_DIR}/${mission_id}.txt"

ZIP_PATH="${ADV_SAMPLE_DIR}/${mission_id}.zip"

# 兼容旧产物，避免历史任务查不到
OLD_TAR_GZ_PATH="${ADV_SAMPLE_DIR}/Attack_generation_inception_v3_${mission_id}.tar.gz"
OLD_TAR_PATH="${ADV_SAMPLE_DIR}/Attack_generation_inception_v3_${mission_id}.tar"

LATEST_LOG_FILE="${LOG_DIR}/run_${mission_id}_latest.log"

# 当前 run.sh 的 work 目录格式：
# /app/work/${mission_id}_${timestamp}_${pid}
WORK_DIR_GLOB="${WORK_ROOT}/${mission_id}_"*

# 当前 run.sh 的对抗样本目录名：
OUTPUT_DIR_NAME="Attack_generation_inception_v3_${mission_id}"

# ===============================
# 解析 poll 文件中的训练进度
# ===============================
parse_poll_file() {
    local poll_file="$1"

    if [ ! -f "$poll_file" ]; then
        echo "null|null|null|null|null|null"
        return 0
    fi

    python3 - "$poll_file" <<'PY'
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

IFS='|' read -r epoch loss_d loss_g loss_adv loss_g_gan loss_hinge < <(parse_poll_file "$POLL_FILE")

# ===============================
# PID 检查
# ===============================
is_pid_running() {
    local pid="$1"

    if [ -z "$pid" ]; then
        return 1
    fi

    if ! [[ "$pid" =~ ^[0-9]+$ ]]; then
        return 1
    fi

    ps -p "$pid" >/dev/null 2>&1
}

has_pid_file=0
main_running=0
runner_running=0

if [ -f "$PID_FILE" ]; then
    has_pid_file=1
    main_pid=$(cat "$PID_FILE" 2>/dev/null || true)
    if is_pid_running "$main_pid"; then
        main_running=1
    fi
fi

if [ -f "$RUNNER_PID_FILE" ]; then
    has_pid_file=1
    runner_pid=$(cat "$RUNNER_PID_FILE" 2>/dev/null || true)
    if is_pid_running "$runner_pid"; then
        runner_running=1
    fi
fi

# ===============================
# 中间目录 / 最终产物检查
# ===============================
has_work_dir=0
has_adv_folder=0
has_zip=0
has_eval_result=0
has_poll_file=0
has_log_file=0

latest_work_dir=""

if compgen -G "$WORK_DIR_GLOB" >/dev/null 2>&1; then
    has_work_dir=1
    latest_work_dir=$(ls -dt $WORK_DIR_GLOB 2>/dev/null | head -n 1)
fi

if [ -n "$latest_work_dir" ]; then
    if [ -d "${latest_work_dir}/results/examples/${OUTPUT_DIR_NAME}" ]; then
        has_adv_folder=1
        ADV_FOLDER="${latest_work_dir}/results/examples/${OUTPUT_DIR_NAME}"
    fi
fi

if [ "$has_adv_folder" -eq 0 ]; then
    found_adv_folder=$(find "$WORK_ROOT" -maxdepth 5 -type d -name "$OUTPUT_DIR_NAME" 2>/dev/null | sort | tail -n 1)
    if [ -n "$found_adv_folder" ] && [ -d "$found_adv_folder" ]; then
        has_adv_folder=1
        ADV_FOLDER="$found_adv_folder"
    fi
fi

if [ -f "$ZIP_PATH" ] || [ -f "$OLD_TAR_GZ_PATH" ] || [ -f "$OLD_TAR_PATH" ]; then
    has_zip=1
fi

if [ -f "$EVAL_RESULT_FILE" ]; then
    has_eval_result=1
fi

if [ -f "$POLL_FILE" ]; then
    has_poll_file=1
fi

if [ -f "$LATEST_LOG_FILE" ]; then
    has_log_file=1
fi

# ===============================
# 统计 Datanum
# ===============================
count_from_zip() {
    local zip_file="$1"

    if [ ! -f "$zip_file" ]; then
        echo "0"
        return 0
    fi

    if command -v unzip >/dev/null 2>&1; then
        unzip -Z1 "$zip_file" 2>/dev/null | grep -E '\.png$' | wc -l | tr -d ' '
        return 0
    fi

    echo "0"
}

count_from_adv_folder() {
    local folder="$1"

    if [ ! -d "$folder" ]; then
        echo "0"
        return 0
    fi

    find "$folder" -maxdepth 1 -type f -name "*.png" 2>/dev/null | wc -l | tr -d ' '
}

if [ -f "$ZIP_PATH" ]; then
    datanum=$(count_from_zip "$ZIP_PATH")
elif [ "$has_adv_folder" -eq 1 ]; then
    datanum=$(count_from_adv_folder "$ADV_FOLDER")
else
    datanum="0"
fi

if [ -z "$datanum" ]; then
    datanum="0"
fi

# ===============================
# 状态判断
# ===============================

# 1. 正在执行中
# main_test.py 或 runner 还在跑，都认为任务执行中
if [ "$main_running" -eq 1 ] || [ "$runner_running" -eq 1 ]; then
    json_running "$datanum" "$epoch" "$loss_d" "$loss_g" "$loss_adv" "$loss_g_gan" "$loss_hinge"
    exit 0
fi

# 2. 已结束 / 查询成功
# 新逻辑优先判断 /app/adv_sample/${mission_id}.zip
# 同时兼容旧 tar/tar.gz
if [ "$has_zip" -eq 1 ]; then
    json_success_done "$datanum" "$epoch" "$loss_d" "$loss_g" "$loss_adv" "$loss_g_gan" "$loss_hinge"
    exit 0
fi

# 3. 任务失败
# 有 eval 结果但没有最终压缩包，说明核心任务可能结束但打包失败
if [ "$has_eval_result" -eq 1 ] && [ "$has_zip" -eq 0 ]; then
    json_task_failed "已有评估结果，但未生成最终压缩包"
    exit 0
fi

# 4. 任务失败
# 有中间对抗样本目录但没有最终压缩包，说明生成后打包或后处理失败
if [ "$has_adv_folder" -eq 1 ] && [ "$has_zip" -eq 0 ]; then
    json_task_failed "已有中间对抗样本目录，但未生成最终压缩包"
    exit 0
fi

# 5. 任务失败
# 曾经启动过，有 PID 文件，但现在没有进程，也没有最终压缩包
if [ "$has_pid_file" -eq 1 ] && \
   [ "$main_running" -eq 0 ] && \
   [ "$runner_running" -eq 0 ] && \
   [ "$has_zip" -eq 0 ]; then
    json_task_failed "任务进程已结束，但未生成最终压缩包"
    exit 0
fi

# 6. 任务失败
# 有训练进度 poll 或日志，但没有进程、没有产物
if { [ "$has_poll_file" -eq 1 ] || [ "$has_log_file" -eq 1 ] || [ "$has_work_dir" -eq 1 ]; } && \
   [ "$has_zip" -eq 0 ]; then
    json_task_failed "任务存在运行痕迹，但未生成最终压缩包"
    exit 0
fi

# 7. 任务不存在
# 没有 PID、没有 work、没有 zip、没有 eval、没有 poll、没有日志
if [ "$has_pid_file" -eq 0 ] && \
   [ "$has_work_dir" -eq 0 ] && \
   [ "$has_adv_folder" -eq 0 ] && \
   [ "$has_zip" -eq 0 ] && \
   [ "$has_eval_result" -eq 0 ] && \
   [ "$has_poll_file" -eq 0 ] && \
   [ "$has_log_file" -eq 0 ]; then
    json_not_exist
    exit 0
fi

# 8. 兜底
json_not_exist
exit 0