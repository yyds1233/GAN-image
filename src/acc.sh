#!/bin/bash

# =========================================
# ACC 启动脚本：JSON 入参 + 参数检查 + seed/weight 检查 + 异步执行 + 立即 JSON 返回 + 日志记录
#
# 启动方式：
#   ./acc.sh '{"mission_id":"2059839135698386944","model_name":"Inception_v3","model_class":"image","method":"ACC"}'
#
# 当前核心执行逻辑不改：
#   python3 -u /app/src/calc_seed_acc.py "${mission_id}" "${weight_arg}"
#
# seed 处理：
#   必须存在 /app/seed/${mission_id}.zip
#   找不到直接返回 400，不等待
#
# weight 处理：
#   如果存在 /app/weight/${mission_id}.zip，则使用上传权重
#   如果不存在，则使用默认权重：
#       /app/weight/inception_v3_google-0cc3c7bd
#       /app/weight/inception_v3_google-0cc3c7bd.pth
#       /app/weight/inception_v3_google-0cc3c7bd.pt
#
# 结果文件：
#   /app/adv_eval/eval_${mission_id}.txt
#   /app/ACC_result/ACC_${mission_id}.txt
#
# 打包结果：
#   /app/ACC_result/${mission_id}.zip
#
# zip 内部结构：
#   ACC_${mission_id}/
#   ├── ACC_${mission_id}.txt
#   ├── eval_${mission_id}.txt
#   └── ACC_${mission_id}.log
#
# 日志文件：
#   /app/run_logs/run_acc_${mission_id}_${timestamp}.log
#   /app/run_logs/run_acc_${mission_id}_latest.log
#
# PID 文件：
#   /tmp/eval_acc_task_${mission_id}.pid   后台 runner pid
#   /tmp/eval_acc_${mission_id}.pid        calc_seed_acc.py pid
#   /tmp/acc_pid_${mission_id}             兼容旧 acc_poll.sh 的 pid 文件
# =========================================

SILENT_MODE=True

# ===============================
# JSON 输出函数
# ===============================
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

json_param_error() {
    echo "{
    \"code\": 400,
    \"message\": \"任务失败\",
    \"data\": {
        \"status\": \"3\",
        \"msg\": \"参数输入错误\"
    }
}"
}

json_file_error() {
    local msg="$1"

    MSG="${msg}" python3 - <<'PY'
import json
import os

print(json.dumps({
    "code": 400,
    "message": "任务失败",
    "data": {
        "status": "3",
        "msg": os.environ["MSG"]
    }
}, ensure_ascii=False, indent=4))
PY
}

fail_response() {
    json_param_error
    exit 1
}

# ===============================
# 工具函数
# ===============================
json_get() {
    local key="$1"
    local default_value="${2:-None}"

    JSON_INPUT="${json_input}" python3 - "$key" "$default_value" <<'PY' 2>/dev/null
import json
import os
import sys

key = sys.argv[1]
default_value = sys.argv[2]

try:
    data = json.loads(os.environ.get("JSON_INPUT", ""))
    value = data.get(key, default_value)
    if value is None:
        value = default_value
    print(value)
except Exception:
    print(default_value)
PY
}

validate_safe_token() {
    local name="$1"
    local value="$2"

    if ! [[ "${value}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
        echo "invalid ${name}: ${value}" >> "${LOG_FILE:-/dev/null}" 2>/dev/null || true
        fail_response
    fi
}

require_command() {
    local cmd="$1"
    command -v "${cmd}" >/dev/null 2>&1 || {
        json_file_error "required command not found: ${cmd}"
        exit 1
    }
}

resolve_default_weight_path_for_check() {
    local p
    for p in \
        "${DEFAULT_WEIGHT_STEM}" \
        "${DEFAULT_WEIGHT_STEM}.pth" \
        "${DEFAULT_WEIGHT_STEM}.pt"
    do
        if [ -f "${p}" ]; then
            printf '%s\n' "${p}"
            return 0
        fi
    done

    return 1
}

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

init_log() {
    LOG_DIR="/app/run_logs"
    mkdir -p "$LOG_DIR"

    RUN_TS=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="${LOG_DIR}/run_acc_${mission_id}_${RUN_TS}.log"
    LATEST_LOG_FILE="${LOG_DIR}/run_acc_${mission_id}_latest.log"

    touch "$LOG_FILE"
    ln -sfn "$LOG_FILE" "$LATEST_LOG_FILE"

    {
        echo "============================================================"
        echo "ACC run log started"
        echo "mission_id: ${mission_id}"
        echo "timestamp: ${RUN_TS}"
        echo "log_file: ${LOG_FILE}"
        echo "latest_log_file: ${LATEST_LOG_FILE}"
        echo "SILENT_MODE: ${SILENT_MODE}"
        echo "============================================================"
    } >> "$LOG_FILE"
}

log_msg() {
    local msg="$1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [mission_id=${mission_id}] [ACC] ${msg}" >> "$LOG_FILE"
}

# ===============================
# 1. 参数解析
# ===============================
if [ "$#" -ne 1 ]; then
    fail_response
fi

json_input="$1"

mission_id="$(json_get "mission_id" "None")"
test_model="$(json_get "model_name" "None")"
model_class="$(json_get "model_class" "None")"
eval_method="$(json_get "method" "None")"

# 可选参数
batch_size="$(json_get "batch_size" "32")"
num_workers="$(json_get "num_workers" "0")"
keep_work="$(json_get "keep_work" "False")"

if [ -z "$mission_id" ] || [ "$mission_id" = "None" ]; then
    fail_response
fi

validate_safe_token "mission_id" "$mission_id"

# mission_id 合法后初始化日志
init_log

{
    echo "json_input: ${json_input}"
    echo "parsed mission_id: ${mission_id}"
    echo "parsed test_model: ${test_model}"
    echo "parsed model_class: ${model_class}"
    echo "parsed eval_method: ${eval_method}"
    echo "parsed batch_size: ${batch_size}"
    echo "parsed num_workers: ${num_workers}"
    echo "parsed keep_work: ${keep_work}"
} >> "$LOG_FILE"

# ===============================
# 2. 参数检查
# ===============================
if [ -z "$test_model" ] || [ "$test_model" = "None" ]; then
    log_msg "参数检查失败: model_name 为空"
    fail_response
fi

if [ -z "$model_class" ] || [ "$model_class" = "None" ]; then
    log_msg "参数检查失败: model_class 为空"
    fail_response
fi

if [ -z "$eval_method" ] || [ "$eval_method" = "None" ]; then
    log_msg "参数检查失败: method 为空"
    fail_response
fi

case "$test_model" in
    "Inception_v3"|"inception_v3"|"Inception"|"inception")
        CANONICAL_MODEL_NAME="inception_v3"
        ;;
    *)
        log_msg "参数检查失败: 不支持的 model_name=${test_model}"
        fail_response
        ;;
esac

if [ "$model_class" != "image" ]; then
    log_msg "参数检查失败: model_class=${model_class}, 当前只支持 image"
    fail_response
fi

case "$eval_method" in
    "ACC"|"acc")
        CANONICAL_METHOD_NAME="ACC"
        ;;
    *)
        log_msg "参数检查失败: method=${eval_method}, 当前只支持 ACC"
        fail_response
        ;;
esac

if ! [[ "$batch_size" =~ ^[0-9]+$ ]] || [ "$batch_size" -le 0 ]; then
    log_msg "参数检查失败: batch_size=${batch_size}"
    fail_response
fi

if ! [[ "$num_workers" =~ ^[0-9]+$ ]]; then
    log_msg "参数检查失败: num_workers=${num_workers}"
    fail_response
fi

case "$keep_work" in
    "True"|"true"|"1"|"yes"|"Y"|"y")
        KEEP_WORK_FLAG="--keep-work"
        ;;
    *)
        KEEP_WORK_FLAG=""
        ;;
esac

# ===============================
# 3. 路径定义与依赖检查
# ===============================
APP_DIR="/app"
SRC_DIR="${APP_DIR}/src"
SEED_ROOT="${APP_DIR}/seed"
WEIGHT_ROOT="${APP_DIR}/weight"
ADV_EVAL_DIR="${APP_DIR}/adv_eval"
ACC_RESULT_DIR="${APP_DIR}/ACC_result"
WORK_ROOT="${APP_DIR}/work"
PY_SCRIPT="${SRC_DIR}/calc_seed_acc.py"

DEFAULT_WEIGHT_STEM="${WEIGHT_ROOT}/inception_v3_google-0cc3c7bd"

SEED_ZIP="${SEED_ROOT}/${mission_id}.zip"
WEIGHT_ZIP="${WEIGHT_ROOT}/${mission_id}.zip"

TASK_PID_FILE="/tmp/eval_acc_task_${mission_id}.pid"
EVAL_PID_FILE="/tmp/eval_acc_${mission_id}.pid"
LEGACY_ACC_PID_FILE="/tmp/acc_pid_${mission_id}"

EVAL_TXT="${ADV_EVAL_DIR}/eval_${mission_id}.txt"
ACC_DETAIL_TXT="${ACC_RESULT_DIR}/ACC_${mission_id}.txt"
ACC_LOG_TXT="${ACC_RESULT_DIR}/ACC_${mission_id}.log"

ACC_PACKAGE_DIR="${ACC_RESULT_DIR}/ACC_${mission_id}"
ACC_ZIP_PATH="${ACC_RESULT_DIR}/${mission_id}.zip"

require_command python3
require_command unzip
require_command zip

if [ ! -d "$SRC_DIR" ]; then
    log_msg "src 目录不存在: ${SRC_DIR}"
    json_file_error "src 目录不存在"
    exit 1
fi

if [ ! -f "$PY_SCRIPT" ]; then
    log_msg "calc_seed_acc.py 不存在: ${PY_SCRIPT}"
    json_file_error "calc_seed_acc.py 不存在"
    exit 1
fi

mkdir -p \
    "$SEED_ROOT" \
    "$WEIGHT_ROOT" \
    "$ADV_EVAL_DIR" \
    "$ACC_RESULT_DIR" \
    "$WORK_ROOT" \
    || {
        log_msg "目录初始化失败"
        json_file_error "目录初始化失败"
        exit 1
    }

# ===============================
# 4. 文件存在性检查：seed 不等待，找不到直接 400
# ===============================
{
    echo "seed_zip: ${SEED_ZIP}"
    echo "weight_zip: ${WEIGHT_ZIP}"
    echo "acc_package_dir: ${ACC_PACKAGE_DIR}"
    echo "acc_zip_path: ${ACC_ZIP_PATH}"
} >> "$LOG_FILE"

if [ ! -f "$SEED_ZIP" ]; then
    log_msg "seed 文件不存在: ${SEED_ZIP}"
    json_file_error "seed 文件不存在"
    exit 1
fi

# weight 可选：
# - 如果 /app/weight/${mission_id}.zip 存在，传 mission_id 给 calc_seed_acc.py，让它使用上传权重
# - 如果不存在，传 None 给 calc_seed_acc.py，让它使用默认权重
if [ -f "$WEIGHT_ZIP" ]; then
    WEIGHT_ARG="${mission_id}"
    log_msg "发现上传权重 zip: ${WEIGHT_ZIP}"
else
    WEIGHT_ARG="None"
    if ! DEFAULT_WEIGHT_PATH_CHECK="$(resolve_default_weight_path_for_check)"; then
        log_msg "默认权重不存在: ${DEFAULT_WEIGHT_STEM}(.pth/.pt)"
        json_file_error "默认权重不存在"
        exit 1
    fi
    log_msg "未发现上传权重 zip，使用默认权重: ${DEFAULT_WEIGHT_PATH_CHECK}"
fi

# ===============================
# 5. 防止同 mission 重复启动
# ===============================
if [ -f "$TASK_PID_FILE" ]; then
    old_task_pid="$(cat "$TASK_PID_FILE" 2>/dev/null || true)"
    if is_pid_running "$old_task_pid"; then
        log_msg "ACC 任务已经在运行，task_pid=${old_task_pid}"
        json_file_error "ACC 任务已经在运行"
        exit 1
    else
        log_msg "发现陈旧 task pid 文件，删除: ${TASK_PID_FILE}"
        rm -f "$TASK_PID_FILE"
    fi
fi

if [ -f "$EVAL_PID_FILE" ]; then
    old_eval_pid="$(cat "$EVAL_PID_FILE" 2>/dev/null || true)"
    if is_pid_running "$old_eval_pid"; then
        log_msg "eval_acc 进程已经在运行，pid=${old_eval_pid}"
        json_file_error "ACC 任务已经在运行"
        exit 1
    else
        log_msg "发现陈旧 eval pid 文件，删除: ${EVAL_PID_FILE}"
        rm -f "$EVAL_PID_FILE"
    fi
fi

if [ -f "$LEGACY_ACC_PID_FILE" ]; then
    old_legacy_pid="$(cat "$LEGACY_ACC_PID_FILE" 2>/dev/null || true)"
    if is_pid_running "$old_legacy_pid"; then
        log_msg "旧 ACC pid 文件对应进程仍在运行，pid=${old_legacy_pid}"
        json_file_error "ACC 任务已经在运行"
        exit 1
    else
        log_msg "发现陈旧 legacy acc pid 文件，删除: ${LEGACY_ACC_PID_FILE}"
        rm -f "$LEGACY_ACC_PID_FILE"
    fi
fi

# ===============================
# 6. 生成后台 runner 脚本
# ===============================
TASK_RUNNER_DIR="/tmp/eval_acc_task_runner"
mkdir -p "$TASK_RUNNER_DIR"
TASK_RUNNER_PATH="${TASK_RUNNER_DIR}/run_acc_${mission_id}.sh"

cat > "$TASK_RUNNER_PATH" <<'EOF'
#!/bin/bash
set +e

SILENT_MODE="$1"
mission_id="$2"
test_model="$3"
model_class="$4"
eval_method="$5"
weight_arg="$6"
batch_size="$7"
num_workers="$8"
keep_work_flag="$9"
LOG_FILE="${10}"
LATEST_LOG_FILE="${11}"
ACC_PACKAGE_DIR="${12}"
ACC_ZIP_PATH="${13}"

APP_DIR="/app"
SRC_DIR="${APP_DIR}/src"
SEED_ROOT="${APP_DIR}/seed"
WEIGHT_ROOT="${APP_DIR}/weight"
ADV_EVAL_DIR="${APP_DIR}/adv_eval"
ACC_RESULT_DIR="${APP_DIR}/ACC_result"
WORK_ROOT="${APP_DIR}/work"

PY_SCRIPT="${SRC_DIR}/calc_seed_acc.py"

SEED_ZIP="${SEED_ROOT}/${mission_id}.zip"
WEIGHT_ZIP="${WEIGHT_ROOT}/${mission_id}.zip"

TASK_PID_FILE="/tmp/eval_acc_task_${mission_id}.pid"
EVAL_PID_FILE="/tmp/eval_acc_${mission_id}.pid"
LEGACY_ACC_PID_FILE="/tmp/acc_pid_${mission_id}"

EVAL_TXT="${ADV_EVAL_DIR}/eval_${mission_id}.txt"
ACC_DETAIL_TXT="${ACC_RESULT_DIR}/ACC_${mission_id}.txt"
ACC_LOG_TXT="${ACC_RESULT_DIR}/ACC_${mission_id}.log"

log_msg() {
    local msg="$1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [mission_id=${mission_id}] [ACC] ${msg}"
}

bg_fail() {
    local reason="$1"
    log_msg "ERROR: ${reason}"
    exit 1
}

cleanup() {
    local status=$?

    rm -f "${EVAL_PID_FILE}"
    rm -f "${LEGACY_ACC_PID_FILE}"

    log_msg "ACC runner 退出，exit_code=${status}"
    log_msg "日志文件: ${LOG_FILE}"

    rm -f "$0"
    exit "${status}"
}

trap cleanup EXIT

if [ "$SILENT_MODE" = "True" ]; then
    exec >> "$LOG_FILE" 2>&1
else
    exec > >(tee -a "$LOG_FILE") 2>&1
fi

echo "$$" > "$TASK_PID_FILE"

log_msg "后台 ACC runner 启动"
log_msg "runner pid=$$"
log_msg "日志文件: ${LOG_FILE}"
log_msg "最新日志软链接: ${LATEST_LOG_FILE}"
log_msg "test_model=${test_model}"
log_msg "model_class=${model_class}"
log_msg "eval_method=${eval_method}"
log_msg "weight_arg=${weight_arg}"
log_msg "batch_size=${batch_size}"
log_msg "num_workers=${num_workers}"
log_msg "keep_work_flag=${keep_work_flag}"
log_msg "ACC_PACKAGE_DIR=${ACC_PACKAGE_DIR}"
log_msg "ACC_ZIP_PATH=${ACC_ZIP_PATH}"

mkdir -p \
    "${ADV_EVAL_DIR}" \
    "${ACC_RESULT_DIR}" \
    "${WORK_ROOT}" \
    || bg_fail "创建输出目录失败"

# ===============================
# 清理旧结果
# ===============================
log_msg "清理旧 ACC 结果文件"
rm -f "${EVAL_TXT}"
rm -f "${ACC_DETAIL_TXT}"
rm -f "${ACC_LOG_TXT}"
rm -f "${ACC_ZIP_PATH}"
rm -rf "${ACC_PACKAGE_DIR}"

# 只清理当前 mission 的 ACC work 临时目录
rm -rf "${WORK_ROOT}/ACC_${mission_id}_"* 2>/dev/null || true

# ===============================
# seed 检查
# ===============================
if [ ! -f "${SEED_ZIP}" ]; then
    bg_fail "seed 文件不存在: ${SEED_ZIP}"
fi

log_msg "seed 文件存在: ${SEED_ZIP}"

# ===============================
# weight 检查
# ===============================
if [ "${weight_arg}" = "None" ] || [ "${weight_arg}" = "none" ]; then
    log_msg "未发现上传权重 zip，calc_seed_acc.py 将使用默认权重"
else
    if [ ! -f "${WEIGHT_ZIP}" ]; then
        bg_fail "weight_arg 非 None，但上传权重 zip 不存在: ${WEIGHT_ZIP}"
    fi
    log_msg "上传权重 zip 存在: ${WEIGHT_ZIP}"
fi

# ===============================
# 核心 ACC 计算逻辑
# 不改核心逻辑，只改为后台 runner 调用
# ===============================
cd "${SRC_DIR}" || bg_fail "无法进入 ${SRC_DIR}"

log_msg "开始执行 calc_seed_acc.py"
log_msg "命令: python3 -u ${PY_SCRIPT} ${mission_id} ${weight_arg} --batch-size ${batch_size} --num-workers ${num_workers} ${keep_work_flag}"

python3 -u "${PY_SCRIPT}" \
    "${mission_id}" \
    "${weight_arg}" \
    --batch-size "${batch_size}" \
    --num-workers "${num_workers}" \
    ${keep_work_flag} &

eval_pid=$!
echo "$eval_pid" > "$EVAL_PID_FILE"
echo "$eval_pid" > "$LEGACY_ACC_PID_FILE"

log_msg "calc_seed_acc.py started, pid=${eval_pid}"
log_msg "eval pid file: ${EVAL_PID_FILE}"
log_msg "legacy acc pid file: ${LEGACY_ACC_PID_FILE}"

wait "$eval_pid"
rc=$?

rm -f "$EVAL_PID_FILE"
rm -f "$LEGACY_ACC_PID_FILE"

log_msg "calc_seed_acc.py finished, exit_code=${rc}"

if [ "$rc" -ne 0 ]; then
    bg_fail "calc_seed_acc.py 执行失败，exit_code=${rc}"
fi

if [ ! -f "${EVAL_TXT}" ]; then
    bg_fail "ACC eval 文件未生成: ${EVAL_TXT}"
fi

if [ ! -f "${ACC_DETAIL_TXT}" ]; then
    bg_fail "ACC detail 文件未生成: ${ACC_DETAIL_TXT}"
fi

# 同步一份 ACC 运行日志到 /app/ACC_result，方便 acc_poll 或人工查看
cp "${LOG_FILE}" "${ACC_LOG_TXT}" 2>/dev/null || true

# ===============================
# 打包 ACC 结果
# 最终产物：
#   /app/ACC_result/${mission_id}.zip
#
# 压缩包内部结构：
#   ACC_${mission_id}/
#   ├── ACC_${mission_id}.txt
#   ├── eval_${mission_id}.txt
#   └── ACC_${mission_id}.log
# ===============================
log_msg "开始整理 ACC 打包目录: ${ACC_PACKAGE_DIR}"

rm -rf "${ACC_PACKAGE_DIR}"
mkdir -p "${ACC_PACKAGE_DIR}" || bg_fail "创建 ACC 打包目录失败: ${ACC_PACKAGE_DIR}"

cp "${ACC_DETAIL_TXT}" "${ACC_PACKAGE_DIR}/ACC_${mission_id}.txt" \
    || bg_fail "复制 ACC detail txt 到打包目录失败"

cp "${EVAL_TXT}" "${ACC_PACKAGE_DIR}/eval_${mission_id}.txt" \
    || bg_fail "复制 eval txt 到打包目录失败"

if [ -f "${ACC_LOG_TXT}" ]; then
    cp "${ACC_LOG_TXT}" "${ACC_PACKAGE_DIR}/ACC_${mission_id}.log" \
        || bg_fail "复制 ACC log 到打包目录失败"
fi

log_msg "开始压缩 ACC 结果: ${ACC_ZIP_PATH}"

rm -f "${ACC_ZIP_PATH}"
tmp_acc_zip="${ACC_ZIP_PATH}.tmp.$$"
rm -f "${tmp_acc_zip}"

(
    cd "${ACC_RESULT_DIR}" || exit 1
    zip -qr "${tmp_acc_zip}" "ACC_${mission_id}"
)

zip_ret=$?
log_msg "zip ACC result exit code: ${zip_ret}"

if [ "${zip_ret}" -ne 0 ]; then
    rm -f "${tmp_acc_zip}"
    bg_fail "压缩 ACC 结果失败: ${tmp_acc_zip}"
fi

mv -f "${tmp_acc_zip}" "${ACC_ZIP_PATH}" \
    || bg_fail "移动 ACC zip 失败: ${ACC_ZIP_PATH}"

if [ ! -f "${ACC_ZIP_PATH}" ]; then
    bg_fail "ACC zip 未生成: ${ACC_ZIP_PATH}"
fi

log_msg "ACC eval txt: ${EVAL_TXT}"
log_msg "ACC detail txt: ${ACC_DETAIL_TXT}"
log_msg "ACC log txt: ${ACC_LOG_TXT}"
log_msg "ACC package dir: ${ACC_PACKAGE_DIR}"
log_msg "ACC zip: ${ACC_ZIP_PATH}"
log_msg "ACC 任务执行完成"

exit 0
EOF

chmod 700 "$TASK_RUNNER_PATH"

# ===============================
# 7. 后台启动任务，立即返回 JSON
# ===============================
if command -v setsid >/dev/null 2>&1; then
    nohup setsid bash "$TASK_RUNNER_PATH" \
        "$SILENT_MODE" \
        "$mission_id" \
        "$CANONICAL_MODEL_NAME" \
        "$model_class" \
        "$CANONICAL_METHOD_NAME" \
        "$WEIGHT_ARG" \
        "$batch_size" \
        "$num_workers" \
        "$KEEP_WORK_FLAG" \
        "$LOG_FILE" \
        "$LATEST_LOG_FILE" \
        "$ACC_PACKAGE_DIR" \
        "$ACC_ZIP_PATH" \
        >> "$LOG_FILE" 2>&1 < /dev/null &
else
    nohup bash "$TASK_RUNNER_PATH" \
        "$SILENT_MODE" \
        "$mission_id" \
        "$CANONICAL_MODEL_NAME" \
        "$model_class" \
        "$CANONICAL_METHOD_NAME" \
        "$WEIGHT_ARG" \
        "$batch_size" \
        "$num_workers" \
        "$KEEP_WORK_FLAG" \
        "$LOG_FILE" \
        "$LATEST_LOG_FILE" \
        "$ACC_PACKAGE_DIR" \
        "$ACC_ZIP_PATH" \
        >> "$LOG_FILE" 2>&1 < /dev/null &
fi

task_pid=$!
echo "$task_pid" > "$TASK_PID_FILE"

{
    echo "task_runner_path: ${TASK_RUNNER_PATH}"
    echo "task_pid_file: ${TASK_PID_FILE}"
    echo "task_pid: ${task_pid}"
    echo "eval_acc_pid_file: ${EVAL_PID_FILE}"
    echo "legacy_acc_pid_file: ${LEGACY_ACC_PID_FILE}"
    echo "eval_txt: ${EVAL_TXT}"
    echo "acc_detail_txt: ${ACC_DETAIL_TXT}"
    echo "acc_package_dir: ${ACC_PACKAGE_DIR}"
    echo "acc_zip_path: ${ACC_ZIP_PATH}"
} >> "$LOG_FILE"

disown "$task_pid" 2>/dev/null || true

json_response 200 "成功" "1"
exit 0