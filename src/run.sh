#!/bin/bash

# =========================================
# GAN Image AdvGAN 启动脚本
#
# 启动方式：
#   ./run.sh '{"mission_id":"2059839135698386944","model_name":"Inception_v3","model_class":"image","method":"AdvGan","epoch":20}'
#
# 说明：
# - 主脚本只负责参数检查、seed/weight 初步检查、生成后台 runner、立即返回 JSON
# - 真正耗时任务在后台 runner 中执行
# - seed 固定读取 /app/seed/${mission_id}.zip，不等待，找不到直接 400
# - weight 如果存在 /app/weight/${mission_id}.zip，则使用上传权重
# - weight 如果不存在，则使用默认 /app/weight/inception_v3_google-0cc3c7bd(.pth/.pt)
# - 核心执行逻辑保持：生成 config 后执行 python3 -u main_test.py --config
# - 最终对抗样本包输出：/app/adv_sample/${mission_id}.zip
# =========================================

SILENT_MODE=True

return_json() {
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
        printf '{\n    "code": %s,\n    "message": "%s",\n    "data": {\n        "status": "%s"\n    }\n}\n' "${code}" "${message}" "${status}"
    fi
}

fail_response() {
    local msg="${1:-参数不合法}"
    return_json 400 "$msg" 3
    exit 1
}

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

is_positive_int() {
    [[ "$1" =~ ^[0-9]+$ ]] && [ "$1" -gt 0 ]
}

is_positive_number() {
    [[ "$1" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || return 1
    awk -v n="$1" 'BEGIN { exit !(n > 0) }'
}

validate_safe_token() {
    local name="$1"
    local value="$2"

    if ! [[ "${value}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
        fail_response "invalid ${name} '${value}'. Allowed characters: A-Z a-z 0-9 _ . -"
    fi
}

require_command() {
    local cmd="$1"
    command -v "${cmd}" >/dev/null 2>&1 || fail_response "required command not found: ${cmd}"
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

init_log() {
    LOG_DIR="/app/run_logs"
    mkdir -p "$LOG_DIR"

    RUN_TS=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="${LOG_DIR}/run_${mission_id}_${RUN_TS}.log"
    LATEST_LOG_FILE="${LOG_DIR}/run_${mission_id}_latest.log"

    touch "$LOG_FILE"
    ln -sfn "$LOG_FILE" "$LATEST_LOG_FILE"

    {
        echo "============================================================"
        echo "GAN Image AdvGAN run log started"
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
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [mission_id=${mission_id}] ${msg}" >> "$LOG_FILE"
}

json_input="${1:-}"

if [ -z "$json_input" ]; then
    return_json 400 "参数不合法" 3
    exit 1
fi

mission_id="$(json_get "mission_id" "None")"
test_model="$(json_get "model_name" "None")"
model_class="$(json_get "model_class" "None")"
test_method="$(json_get "method" "None")"
advgan_epochs="$(json_get "epoch" "None")"

timeout_sec="$(json_get "timeout" "3600")"
l_inf_bound="$(json_get "l_inf_bound" "0.05")"
advgan_lr="$(json_get "advgan_lr" "0.001")"

if [ -z "$mission_id" ] || [ "$mission_id" = "None" ]; then
    return_json 400 "参数不合法" 3
    exit 1
fi

validate_safe_token "mission_id" "$mission_id"
init_log

{
    echo "json_input: ${json_input}"
    echo "parsed mission_id: ${mission_id}"
    echo "parsed test_model: ${test_model}"
    echo "parsed model_class: ${model_class}"
    echo "parsed test_method: ${test_method}"
    echo "parsed advgan_epochs: ${advgan_epochs}"
    echo "parsed timeout_sec: ${timeout_sec}"
    echo "parsed l_inf_bound: ${l_inf_bound}"
    echo "parsed advgan_lr: ${advgan_lr}"
} >> "$LOG_FILE"

if [ -z "$model_class" ] || [ "$model_class" = "None" ] || \
   [ -z "$test_model" ] || [ "$test_model" = "None" ] || \
   [ -z "$test_method" ] || [ "$test_method" = "None" ] || \
   [ -z "$advgan_epochs" ] || [ "$advgan_epochs" = "None" ]; then
    log_msg "参数检查失败"
    return_json 400 "参数不合法" 3
    exit 1
fi

if [ "$model_class" != "image" ]; then
    log_msg "model_class 无效: ${model_class}"
    return_json 400 "model_class 无效" 3
    exit 1
fi

case "$test_model" in
    "Inception_v3"|"inception_v3"|"Inception"|"inception")
        CANONICAL_MODEL_NAME="inception_v3"
        ;;
    *)
        log_msg "model_name 无效: ${test_model}"
        return_json 400 "model_name 无效" 3
        exit 1
        ;;
esac

case "$test_method" in
    "AdvGan"|"AdvGAN"|"advgan")
        CANONICAL_METHOD_NAME="AdvGan"
        ;;
    *)
        log_msg "method 无效: ${test_method}"
        return_json 400 "method 无效" 3
        exit 1
        ;;
esac

if ! is_positive_int "$advgan_epochs"; then
    log_msg "epoch 非正整数: ${advgan_epochs}"
    return_json 400 "epoch 必须是正整数" 3
    exit 1
fi

if ! is_positive_int "$timeout_sec"; then
    log_msg "timeout 非正整数: ${timeout_sec}"
    return_json 400 "timeout 必须是正整数" 3
    exit 1
fi

if ! is_positive_number "$l_inf_bound"; then
    log_msg "l_inf_bound 非正数: ${l_inf_bound}"
    return_json 400 "l_inf_bound 必须是正数" 3
    exit 1
fi

if ! is_positive_number "$advgan_lr"; then
    log_msg "advgan_lr 非正数: ${advgan_lr}"
    return_json 400 "advgan_lr 必须是正数" 3
    exit 1
fi

APP_DIR="/app"
SRC_DIR="${APP_DIR}/src"
SEED_ROOT="${APP_DIR}/seed"
WEIGHT_ROOT="${APP_DIR}/weight"
WORK_ROOT="${APP_DIR}/work"
ADV_SAMPLE_DIR="${APP_DIR}/adv_sample"
ADV_EVAL_DIR="${APP_DIR}/adv_eval"
DEFAULT_WEIGHT_STEM="${WEIGHT_ROOT}/inception_v3_google-0cc3c7bd"

SEED_ZIP="${SEED_ROOT}/${mission_id}.zip"
WEIGHT_ZIP="${WEIGHT_ROOT}/${mission_id}.zip"

require_command python3
require_command awk
require_command timeout
require_command unzip
require_command find
require_command zip

if command -v setsid >/dev/null 2>&1; then
    HAS_SETSID=True
else
    HAS_SETSID=False
fi

if [ ! -d "$SRC_DIR" ]; then
    log_msg "source directory not found: ${SRC_DIR}"
    return_json 400 "src 目录不存在" 3
    exit 1
fi

if [ ! -f "${SRC_DIR}/main_test.py" ]; then
    log_msg "main_test.py not found: ${SRC_DIR}/main_test.py"
    return_json 400 "main_test.py 不存在" 3
    exit 1
fi

mkdir -p \
    "$SEED_ROOT" \
    "$WEIGHT_ROOT" \
    "$WORK_ROOT" \
    "$ADV_SAMPLE_DIR" \
    "$ADV_EVAL_DIR" \
    || {
        log_msg "failed to create application directories"
        return_json 400 "目录初始化失败" 3
        exit 1
    }

log_msg "seed_zip: ${SEED_ZIP}"
if [ ! -f "$SEED_ZIP" ]; then
    log_msg "seed 文件不存在: ${SEED_ZIP}"
    return_json 400 "seed 文件不存在" 3
    exit 1
fi

log_msg "weight_zip: ${WEIGHT_ZIP}"
if [ -f "$WEIGHT_ZIP" ]; then
    log_msg "发现上传权重 zip: ${WEIGHT_ZIP}"
else
    if ! DEFAULT_WEIGHT_PATH_CHECK="$(resolve_default_weight_path_for_check)"; then
        log_msg "默认 Inception_v3 权重不存在: ${DEFAULT_WEIGHT_STEM}(.pth/.pt)"
        return_json 400 "默认权重不存在" 3
        exit 1
    fi
    log_msg "未发现上传权重 zip，使用默认权重: ${DEFAULT_WEIGHT_PATH_CHECK}"
fi

PID_FILE="/tmp/pid_${mission_id}"
RUNNER_PID_FILE="/tmp/gan_runner_${mission_id}.pid"

if [ -f "$PID_FILE" ]; then
    old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [ -n "$old_pid" ] && kill -0 "$old_pid" 2>/dev/null; then
        log_msg "任务已经在运行，pid=${old_pid}"
        return_json 400 "任务已经在运行" 1
        exit 1
    else
        log_msg "发现陈旧 PID 文件，删除: ${PID_FILE}"
        rm -f "$PID_FILE"
    fi
fi

TASK_RUNNER_DIR="/tmp/gan_task_runner"
mkdir -p "$TASK_RUNNER_DIR"
TASK_RUNNER_PATH="${TASK_RUNNER_DIR}/run_gan_${mission_id}.sh"

cat > "$TASK_RUNNER_PATH" <<'EOF_RUNNER'
#!/bin/bash
set +e

SILENT_MODE="$1"
mission_id="$2"
CANONICAL_MODEL_NAME="$3"
CANONICAL_METHOD_NAME="$4"
model_class="$5"
advgan_epochs="$6"
l_inf_bound="$7"
advgan_lr="$8"
timeout_sec="$9"
LOG_FILE="${10}"
LATEST_LOG_FILE="${11}"

APP_DIR="/app"
SRC_DIR="${APP_DIR}/src"
SEED_ROOT="${APP_DIR}/seed"
WEIGHT_ROOT="${APP_DIR}/weight"
WORK_ROOT="${APP_DIR}/work"
ADV_SAMPLE_DIR="${APP_DIR}/adv_sample"
ADV_EVAL_DIR="${APP_DIR}/adv_eval"
DEFAULT_WEIGHT_STEM="${WEIGHT_ROOT}/inception_v3_google-0cc3c7bd"

PID_FILE="/tmp/pid_${mission_id}"
RUNNER_PID_FILE="/tmp/gan_runner_${mission_id}.pid"

log_msg() {
    local msg="$1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [mission_id=${mission_id}] ${msg}"
}

cleanup() {
    local status=$?

    rm -f "${PID_FILE}"

    if [ -n "${TMP_ZIP_PATH:-}" ]; then
        rm -f "${TMP_ZIP_PATH}"
    fi

    log_msg "任务退出，exit code: ${status}"
    if [ -n "${WORK_DIR:-}" ]; then
        log_msg "Work directory kept: ${WORK_DIR}"
    fi

    rm -f "$0"
    exit "${status}"
}

if [ "$SILENT_MODE" = "True" ]; then
    exec >> "$LOG_FILE" 2>&1
else
    exec > >(tee -a "$LOG_FILE") 2>&1
fi

echo "$$" > "$PID_FILE"

log_msg "日志文件: ${LOG_FILE}"
log_msg "最新日志软链接: ${LATEST_LOG_FILE}"
log_msg "后台 runner 启动"
log_msg "runner pid: $$"
log_msg "model_name: ${CANONICAL_MODEL_NAME}"
log_msg "model_class: ${model_class}"
log_msg "method: ${CANONICAL_METHOD_NAME}"
log_msg "epoch: ${advgan_epochs}"
log_msg "timeout_sec: ${timeout_sec}"
log_msg "l_inf_bound: ${l_inf_bound}"
log_msg "advgan_lr: ${advgan_lr}"

resolve_default_weight_path() {
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

find_dataset_files() {
    local dataset_root="$1"

    IMAGE_DIR="$(find "${dataset_root}" -type d -name "img" | head -n 1 || true)"
    LABEL_CSV="$(find "${dataset_root}" -type f -name "images.csv" | head -n 1 || true)"

    if [ -z "${IMAGE_DIR}" ] || [ ! -d "${IMAGE_DIR}" ]; then
        log_msg "Image directory 'img' not found under ${dataset_root}"
        exit 1
    fi

    if [ -z "${LABEL_CSV}" ] || [ ! -f "${LABEL_CSV}" ]; then
        log_msg "Label file 'images.csv' not found under ${dataset_root}"
        exit 1
    fi

    log_msg "Image dir: ${IMAGE_DIR}"
    log_msg "Label CSV: ${LABEL_CSV}"
}

prepare_seed() {
    SEED_ZIP="${SEED_ROOT}/${mission_id}.zip"

    log_msg "开始处理 seed: ${SEED_ZIP}"

    if [ ! -f "${SEED_ZIP}" ]; then
        log_msg "seed 文件不存在: ${SEED_ZIP}"
        exit 1
    fi

    rm -rf "${SEED_EXTRACT_DIR}"
    mkdir -p "${SEED_EXTRACT_DIR}"

    unzip -q "${SEED_ZIP}" -d "${SEED_EXTRACT_DIR}"
    unzip_ret=$?
    log_msg "unzip seed exit code: ${unzip_ret}"

    if [ "${unzip_ret}" -ne 0 ]; then
        log_msg "seed 解压失败: ${SEED_ZIP}"
        exit 1
    fi

    find_dataset_files "${SEED_EXTRACT_DIR}"
}

prepare_weight() {
    TARGET_WEIGHT_PATH=""
    WEIGHT_ZIP="${WEIGHT_ROOT}/${mission_id}.zip"

    rm -rf "${WEIGHT_EXTRACT_DIR}"
    mkdir -p "${WEIGHT_EXTRACT_DIR}"

    if [ -f "${WEIGHT_ZIP}" ]; then
        log_msg "发现上传权重 zip: ${WEIGHT_ZIP}"
        unzip -q "${WEIGHT_ZIP}" -d "${WEIGHT_EXTRACT_DIR}"
        unzip_ret=$?
        log_msg "unzip weight exit code: ${unzip_ret}"

        if [ "${unzip_ret}" -ne 0 ]; then
            log_msg "weight 解压失败: ${WEIGHT_ZIP}"
            exit 1
        fi

        TARGET_WEIGHT_PATH="$(find "${WEIGHT_EXTRACT_DIR}" -type f \( -name "*.pth" -o -name "*.pt" \) | head -n 1 || true)"

        if [ -z "${TARGET_WEIGHT_PATH}" ]; then
            log_msg "Weight zip exists but no .pth or .pt file found: ${WEIGHT_ZIP}"
            exit 1
        fi

        log_msg "Target uploaded weight: ${TARGET_WEIGHT_PATH}"
    else
        log_msg "未发现上传权重 zip，使用默认 Inception_v3 权重: ${DEFAULT_WEIGHT_STEM}(.pth/.pt)"

        if ! TARGET_WEIGHT_PATH="$(resolve_default_weight_path)"; then
            log_msg "Default weight file not found."
            log_msg "Tried:"
            log_msg "  ${DEFAULT_WEIGHT_STEM}"
            log_msg "  ${DEFAULT_WEIGHT_STEM}.pth"
            log_msg "  ${DEFAULT_WEIGHT_STEM}.pt"
            exit 1
        fi

        log_msg "Target default weight: ${TARGET_WEIGHT_PATH}"
    fi
}

worker_main() {
    trap cleanup EXIT

    log_msg "Mission ${mission_id} accepted."
    log_msg "Run ID: ${RUN_ID}"
    log_msg "Work directory: ${WORK_DIR}"

    rm -f "${ZIP_PATH}"
    rm -f "${TMP_ZIP_PATH}"
    rm -f "${EVAL_TXT}"
    rm -f "${POLL_TXT}"

    prepare_seed
    prepare_weight

    log_msg "Generating mission config: ${CONFIG_PATH}"

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
        log_msg "Config file was not generated: ${CONFIG_PATH}"
        exit 1
    fi

    log_msg "Running mission ${mission_id}..."
    log_msg "Config file: ${CONFIG_PATH}"

    cd "${SRC_DIR}" || exit 1

    set +e
    timeout --kill-after=30s "${timeout_sec}" \
        python3 -u main_test.py --config "${CONFIG_PATH}" &

    core_pid=$!
    echo "${core_pid}" > "${PID_FILE}"
    log_msg "main_test.py pid: ${core_pid}"

    wait "${core_pid}"
    exit_code=$?
    set +e

    log_msg "main_test.py exit code: ${exit_code}"

    if [ "${exit_code}" -eq 124 ]; then
        log_msg "Timeout reached. Mission ${mission_id} stopped."
        exit 124
    elif [ "${exit_code}" -ne 0 ]; then
        log_msg "Mission ${mission_id} failed with exit code ${exit_code}."
        exit "${exit_code}"
    fi

    if [ ! -d "${ADV_IMAGE_DIR}" ]; then
        log_msg "Adversarial image directory not found: ${ADV_IMAGE_DIR}"
        exit 1
    fi

    if ! find "${ADV_IMAGE_DIR}" -type f -name "*.png" | grep -q .; then
        log_msg "No adversarial PNG images generated in ${ADV_IMAGE_DIR}"
        exit 1
    fi

    if [ ! -f "${EVAL_TXT}" ]; then
        log_msg "Evaluation txt not found: ${EVAL_TXT}"
        exit 1
    fi

    if [ ! -f "${ADV_IMAGE_DIR}/ssim.txt" ]; then
        log_msg "ssim.txt not found: ${ADV_IMAGE_DIR}/ssim.txt"
        exit 1
    fi

    if [ ! -f "${ADV_IMAGE_DIR}/value.txt" ]; then
        log_msg "value.txt not found: ${ADV_IMAGE_DIR}/value.txt"
        exit 1
    fi

    log_msg "Compressing adversarial images to zip..."

    rm -f "${ZIP_PATH}"
    rm -f "${TMP_ZIP_PATH}"

    (
        cd "$(dirname "${ADV_IMAGE_DIR}")" || exit 1
        zip -qr "${TMP_ZIP_PATH}" "${OUTPUT_DIR_NAME}"
    )

    zip_ret=$?
    log_msg "zip exit code: ${zip_ret}"

    if [ "${zip_ret}" -ne 0 ]; then
        log_msg "压缩失败: ${TMP_ZIP_PATH}"
        exit 1
    fi

    mv -f "${TMP_ZIP_PATH}" "${ZIP_PATH}"
    mv_ret=$?
    log_msg "move zip exit code: ${mv_ret}"

    if [ "${mv_ret}" -ne 0 ]; then
        log_msg "移动压缩包失败: ${ZIP_PATH}"
        exit 1
    fi

    log_msg "download adv_sample ${ZIP_NAME} in ${ADV_SAMPLE_DIR}"
    log_msg "download adv_eval ${mission_id}.txt in ${ADV_EVAL_DIR}"
    log_msg "Mission ${mission_id} completed successfully."
}

OUTPUT_BASENAME="Attack_generation_${CANONICAL_MODEL_NAME}_${mission_id}"
OUTPUT_DIR_NAME="${OUTPUT_BASENAME}"

RUN_ID="${mission_id}_$(date +%Y%m%d%H%M%S)_$$"
WORK_DIR="${WORK_ROOT}/${RUN_ID}"

SEED_EXTRACT_DIR="${WORK_DIR}/seed"
WEIGHT_EXTRACT_DIR="${WORK_DIR}/weight"
CONFIG_DIR="${WORK_DIR}/config"
CHECKPOINT_DIR="${WORK_DIR}/checkpoints/AdvGAN"
LOSS_DIR="${WORK_DIR}/results/losses"
ADV_IMAGE_DIR="${WORK_DIR}/results/examples/${OUTPUT_DIR_NAME}"
NPY_DIR="${WORK_DIR}/npy"
INNER_LOG_DIR="${WORK_DIR}/logs"

CONFIG_PATH="${CONFIG_DIR}/hyperparams_${mission_id}.json"
EVAL_TXT="${ADV_EVAL_DIR}/${mission_id}.txt"
POLL_TXT="${ADV_EVAL_DIR}/poll_${mission_id}.txt"

ZIP_NAME="${mission_id}.zip"
ZIP_PATH="${ADV_SAMPLE_DIR}/${ZIP_NAME}"
TMP_ZIP_PATH="${ZIP_PATH}.tmp.$$"

mkdir -p \
    "${SEED_EXTRACT_DIR}" \
    "${WEIGHT_EXTRACT_DIR}" \
    "${CONFIG_DIR}" \
    "${CHECKPOINT_DIR}" \
    "${LOSS_DIR}" \
    "${ADV_IMAGE_DIR}" \
    "${NPY_DIR}" \
    "${INNER_LOG_DIR}" \
    "${ADV_SAMPLE_DIR}" \
    "${ADV_EVAL_DIR}" \
    || {
        log_msg "failed to create work directories: ${WORK_DIR}"
        exit 1
    }

ln -sfn "${LOG_FILE}" "${INNER_LOG_DIR}/run.log"

worker_main
exit 0
EOF_RUNNER

chmod 700 "$TASK_RUNNER_PATH"

if [ "$HAS_SETSID" = "True" ]; then
    nohup setsid bash "$TASK_RUNNER_PATH" \
        "$SILENT_MODE" \
        "$mission_id" \
        "$CANONICAL_MODEL_NAME" \
        "$CANONICAL_METHOD_NAME" \
        "$model_class" \
        "$advgan_epochs" \
        "$l_inf_bound" \
        "$advgan_lr" \
        "$timeout_sec" \
        "$LOG_FILE" \
        "$LATEST_LOG_FILE" \
        >> "$LOG_FILE" 2>&1 < /dev/null &
else
    nohup bash "$TASK_RUNNER_PATH" \
        "$SILENT_MODE" \
        "$mission_id" \
        "$CANONICAL_MODEL_NAME" \
        "$CANONICAL_METHOD_NAME" \
        "$model_class" \
        "$advgan_epochs" \
        "$l_inf_bound" \
        "$advgan_lr" \
        "$timeout_sec" \
        "$LOG_FILE" \
        "$LATEST_LOG_FILE" \
        >> "$LOG_FILE" 2>&1 < /dev/null &
fi

runner_pid=$!

echo "$runner_pid" > "$RUNNER_PID_FILE"
echo "$runner_pid" > "$PID_FILE"

disown "$runner_pid" 2>/dev/null || true

log_msg "runner_pid: ${runner_pid}"
log_msg "task_runner_path: ${TASK_RUNNER_PATH}"
log_msg "pid_file: ${PID_FILE}"

return_json 200 "参数合法" 1
exit 0