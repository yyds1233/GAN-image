#!/bin/bash

# =========================================
# ACC 轮询脚本
#
# 用法:
#   ./acc_poll.sh '{"mission_id":"2059839135698386944"}'
#
# 输出情况：
# 1. 参数不正确：
#    code=400, message=查询失败, status=3, msg=参数输入错误
#
# 2. 查询的 id 已经结束，被执行过，可以查到 ACC：
#    有 /app/adv_eval/eval_${mission_id}.txt
#    有 /app/ACC_result/ACC_${mission_id}.txt
#    有 /app/ACC_result/${mission_id}.zip
#    ACC 从 /app/adv_eval/eval_${mission_id}.txt 读取
#    code=200, message=查询成功, ACC=实际值, status=2
#
# 3. 查询的 id 正在执行中：
#    eval_acc_task / eval_acc / legacy acc pid 任意一个正在运行
#    code=200, message=任务正在执行中, ACC=null, status=1
#
# 4. 查询的 id 任务失败：
#    有中间结果但没有完整结果或没有 zip
#    或者曾经启动过但 pid 已结束且没有完整结果
#    code=200, message=任务失败, ACC=null, status=3
#
# 5. 查询的 id 未被启动过：
#    没有 pid file，也没有上述结果文件
#    code=1002, message=任务不存在, data={}
# =========================================

json_input="${1:-}"

# =========================
# JSON 输出函数
# =========================

json_param_error() {
    echo "{
    \"code\": 400,
    \"message\": \"查询失败\",
    \"data\": {
        \"status\": \"3\",
        \"msg\": \"参数输入错误\"
    }
}"
}

json_success_done() {
    local acc="$1"
    local download_result_addr="$2"

    echo "{
    \"code\": 200,
    \"message\": \"查询成功\",
    \"data\": {
        \"ACC\": \"$acc\",
        \"status\": \"2\"
    }
}"
}

json_running() {
    echo "{
    \"code\": 200,
    \"message\": \"任务正在执行中\",
    \"data\": {
        \"ACC\": \"null\",
        \"status\": \"1\"
    }
}"
}

json_task_failed() {
    local msg="${1:-任务失败}"

    echo "{
    \"code\": 200,
    \"message\": \"任务失败\",
    \"data\": {
        \"ACC\": \"null\",
        \"msg\": \"$msg\",
        \"status\": \"3\"
    }
}"
}

json_not_exist() {
    echo "{
    \"code\": 1002,
    \"message\": \"任务不存在\",
    \"data\": {
    }
}"
}

# =========================
# 1. 参数解析
# =========================

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

# =========================
# 2. 参数检查
# =========================

if [ -z "$mission_id" ] || [ "$mission_id" = "None" ]; then
    json_param_error
    exit 1
fi

# 与 acc.sh 保持一致：允许字母、数字、下划线、点、中划线
if ! [[ "$mission_id" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    json_param_error
    exit 1
fi

# =========================
# 3. 路径定义
# =========================

APP_DIR="/app"
ADV_EVAL_DIR="${APP_DIR}/adv_eval"
ACC_RESULT_DIR="${APP_DIR}/ACC_result"
RUN_LOG_DIR="${APP_DIR}/run_logs"
WORK_ROOT="${APP_DIR}/work"

# ACC 总结果文件：用于读取 ACC 数值
middle_result_file="${ADV_EVAL_DIR}/eval_${mission_id}.txt"

# ACC 明细结果文件：用于判断任务是否完整结束
final_result_file="${ACC_RESULT_DIR}/ACC_${mission_id}.txt"

# ACC 打包目录和最终 zip：用于判断任务是否完整打包结束
acc_package_dir="${ACC_RESULT_DIR}/ACC_${mission_id}"
acc_zip_file="${ACC_RESULT_DIR}/${mission_id}.zip"

# 日志文件：用于判断任务是否曾经启动过
latest_log_file="${RUN_LOG_DIR}/run_acc_${mission_id}_latest.log"

# ACC work 目录：用于判断任务是否曾经启动过
acc_work_glob="${WORK_ROOT}/ACC_${mission_id}_"*

# PID 文件
pid_file="/tmp/eval_acc_${mission_id}.pid"
task_pid_file="/tmp/eval_acc_task_${mission_id}.pid"
legacy_pid_file="/tmp/acc_pid_${mission_id}"

# =========================
# 4. 判断 pid 是否还在运行
# =========================

is_pid_running() {
    local file="$1"

    if [ ! -f "$file" ]; then
        return 1
    fi

    local pid
    pid=$(cat "$file" 2>/dev/null)

    if ! [[ "$pid" =~ ^[0-9]+$ ]]; then
        return 1
    fi

    if ps -p "$pid" >/dev/null 2>&1; then
        return 0
    fi

    return 1
}

pid_running=0
has_pid_file=0

if [ -f "$pid_file" ] || [ -f "$task_pid_file" ] || [ -f "$legacy_pid_file" ]; then
    has_pid_file=1
fi

if is_pid_running "$pid_file" || \
   is_pid_running "$task_pid_file" || \
   is_pid_running "$legacy_pid_file"; then
    pid_running=1
fi

# =========================
# 5. 检查产物和运行痕迹
# =========================

has_middle_result=0
has_final_result=0
has_zip=0
has_package_dir=0
has_log_file=0
has_work_dir=0

if [ -f "$middle_result_file" ]; then
    has_middle_result=1
fi

if [ -f "$final_result_file" ]; then
    has_final_result=1
fi

if [ -f "$acc_zip_file" ]; then
    has_zip=1
fi

if [ -d "$acc_package_dir" ]; then
    has_package_dir=1
fi

if [ -f "$latest_log_file" ]; then
    has_log_file=1
fi

if compgen -G "$acc_work_glob" >/dev/null 2>&1; then
    has_work_dir=1
fi

# =========================
# 6. 读取 ACC
# =========================

read_acc_value() {
    local file="$1"

    if [ ! -f "$file" ]; then
        echo "null"
        return 0
    fi

    local acc_line
    acc_line="$(head -n 1 "$file" 2>/dev/null)"

    if [ -z "$acc_line" ]; then
        echo "null"
        return 0
    fi

    # 兼容文件内容：
    # ACC: 86.23
    # ACC:86.23
    # 86.23
    acc_value="${acc_line#ACC:}"
    acc_value="$(echo "$acc_value" | xargs)"

    if [ -z "$acc_value" ]; then
        echo "null"
    else
        echo "$acc_value"
    fi
}

# =========================
# 7. 状态判断
# =========================

# 7.1 正在执行中
# 有任一 pid 正在运行，认为任务正在执行
if [ "$pid_running" -eq 1 ]; then
    json_running
    exit 0
fi

# 7.2 已结束 / 查询成功
# 完整完成条件：
#   eval_${mission_id}.txt 存在
#   ACC_${mission_id}.txt 存在
#   ${mission_id}.zip 存在
if [ "$has_middle_result" -eq 1 ] && \
   [ "$has_final_result" -eq 1 ] && \
   [ "$has_zip" -eq 1 ]; then

    acc_value="$(read_acc_value "$middle_result_file")"
    json_success_done "$acc_value" "$acc_zip_file"
    exit 0
fi

# 7.3 任务失败
# 有 ACC 数值文件和明细文件，但是没有 zip，说明计算完成但打包失败
if [ "$has_middle_result" -eq 1 ] && \
   [ "$has_final_result" -eq 1 ] && \
   [ "$has_zip" -eq 0 ]; then
    json_task_failed "ACC 已计算完成，但未生成结果压缩包"
    exit 0
fi

# 7.4 任务失败
# 有 ACC 总结果文件，但是没有明细结果文件
if [ "$has_middle_result" -eq 1 ] && \
   [ "$has_final_result" -eq 0 ]; then
    json_task_failed "已有 ACC 总结果文件，但未生成 ACC 明细结果文件"
    exit 0
fi

# 7.5 任务失败
# 有 ACC 明细文件或打包目录，但是没有总结果文件
if [ "$has_middle_result" -eq 0 ] && \
   { [ "$has_final_result" -eq 1 ] || [ "$has_package_dir" -eq 1 ]; }; then
    json_task_failed "已有 ACC 中间产物，但未生成 ACC 总结果文件"
    exit 0
fi

# 7.6 任务失败兜底
# 曾经启动过，有 pid file，但 pid 已经不运行，并且没有完整结果
if [ "$has_pid_file" -eq 1 ] && \
   [ "$pid_running" -eq 0 ] && \
   { [ "$has_middle_result" -eq 0 ] || [ "$has_final_result" -eq 0 ] || [ "$has_zip" -eq 0 ]; }; then
    json_task_failed "任务进程已结束，但未生成完整 ACC 结果"
    exit 0
fi

# 7.7 任务失败
# 有日志或 work 目录，说明启动过，但没有完整结果
if { [ "$has_log_file" -eq 1 ] || [ "$has_work_dir" -eq 1 ]; } && \
   { [ "$has_middle_result" -eq 0 ] || [ "$has_final_result" -eq 0 ] || [ "$has_zip" -eq 0 ]; }; then
    json_task_failed "任务存在运行痕迹，但未生成完整 ACC 结果"
    exit 0
fi

# 7.8 未被启动过
# 没有 pid、没有结果、没有 zip、没有日志、没有 work
if [ "$has_pid_file" -eq 0 ] && \
   [ "$has_middle_result" -eq 0 ] && \
   [ "$has_final_result" -eq 0 ] && \
   [ "$has_zip" -eq 0 ] && \
   [ "$has_package_dir" -eq 0 ] && \
   [ "$has_log_file" -eq 0 ] && \
   [ "$has_work_dir" -eq 0 ]; then
    json_not_exist
    exit 0
fi

# 7.9 兜底
json_not_exist
exit 0