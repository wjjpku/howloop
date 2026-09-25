#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 4 || "$#" -gt 5 ]]; then
    echo "usage: $0 ACTION TASK SUPERVISION PHYSICAL_GPU [SEED]" >&2
    exit 2
fi

ACTION="$1"
TASK="$2"
SUPERVISION="$3"
PHYSICAL_GPU="$4"
SEED="${5:-0}"
BASELINE_VARIANT="${BASELINE_VARIANT:-}"
CONTROLLER_CURRICULUM="${CONTROLLER_CURRICULUM:-extension}"
CONTROLLER_LOGICAL_MAX_LENGTH="${CONTROLLER_LOGICAL_MAX_LENGTH:-}"
CONTROLLER_LOGICAL_MIN_LENGTH="${CONTROLLER_LOGICAL_MIN_LENGTH:-}"
CONTROLLER_LABEL_OVERRIDE="${CONTROLLER_LABEL_OVERRIDE:-}"
CONTROLLER_SEED="${CONTROLLER_SEED:-211001}"
CONTROLLER_TRAINING_PROFILE="${CONTROLLER_TRAINING_PROFILE:-legacy}"
CONTROLLER_INITIALIZATION="${CONTROLLER_INITIALIZATION:-dense_svd}"
CONTROLLER_FORCE="${CONTROLLER_FORCE:-0}"
CONTROLLER_ANCHOR_STEP="${CONTROLLER_ANCHOR_STEP:-}"
CONTROLLER_WARMUP_UPDATES="${CONTROLLER_WARMUP_UPDATES:-0}"
CONTROLLER_FINAL_LR_RATIO="${CONTROLLER_FINAL_LR_RATIO:-1.0}"
CONTROLLER_LR_SCHEDULE="${CONTROLLER_LR_SCHEDULE:-cosine}"
CONTROLLER_STABLE_UPDATES="${CONTROLLER_STABLE_UPDATES:-0}"
CONTROLLER_POST_FINAL_J="${CONTROLLER_POST_FINAL_J:-0}"
CONTROLLER_DENSE_STAGE_COUNT="${CONTROLLER_DENSE_STAGE_COUNT:-2}"
CONTROLLER_GRAD_CLIP="${CONTROLLER_GRAD_CLIP:-1.0}"
CONTROLLER_LEARNING_RATE_MULTIPLIER="${CONTROLLER_LEARNING_RATE_MULTIPLIER:-1.0}"
CONTROLLER_DIAGONAL_LR_MULTIPLIER="${CONTROLLER_DIAGONAL_LR_MULTIPLIER:-0.1}"
CONTROLLER_CE_TEMPERATURE="${CONTROLLER_CE_TEMPERATURE:-1.0}"
CONTROLLER_STAGE_ROUND_MULTIPLIER="${CONTROLLER_STAGE_ROUND_MULTIPLIER:-1}"
AUDIT_SEED="${AUDIT_SEED:-261001}"
AUDIT_BATCH_SIZE="${AUDIT_BATCH_SIZE:-128}"
AUDIT_BATCHES="${AUDIT_BATCHES:-32}"
AUDIT_LENGTHS="${AUDIT_LENGTHS:-20 40 50 75 84 100}"
AUDIT_MAXIMUM_STEP="${AUDIT_MAXIMUM_STEP:-132}"
AUDIT_MODES="${AUDIT_MODES:-full no_AB identity_D mean_D no_bias shuffle_D}"
AUDIT_POST_FINAL_J="${AUDIT_POST_FINAL_J:-0}"
AUDIT_DIR_OVERRIDE="${AUDIT_DIR_OVERRIDE:-}"
DIAGNOSIS_BATCH_SIZE="${DIAGNOSIS_BATCH_SIZE:-128}"
DIAGNOSIS_BATCHES="${DIAGNOSIS_BATCHES:-32}"
DIAGNOSIS_LENGTHS="${DIAGNOSIS_LENGTHS:-20 40 50 75 100}"
DIAGNOSIS_MAXIMUM_STEP="${DIAGNOSIS_MAXIMUM_STEP:-132}"
DIAGNOSIS_EXTENSION_GATE_LENGTH="${DIAGNOSIS_EXTENSION_GATE_LENGTH:-40}"

if [[ "${ACTION}" != "smoke" && "${ACTION}" != "benchmark" && "${ACTION}" != "released_benchmark" && "${ACTION}" != "released_benchmark_fp32" && "${ACTION}" != "official_benchmark_fp32" && "${ACTION}" != "pilot" && "${ACTION}" != "formal" && "${ACTION}" != "released_formal" && "${ACTION}" != "official_formal" && "${ACTION}" != "diagnose" && "${ACTION}" != "controller" && "${ACTION}" != "audit" ]]; then
    echo "unsupported action: ${ACTION}" >&2
    exit 2
fi
if [[ "${TASK}" != "parity" && "${TASK}" != "copy" && "${TASK}" != "copy4" && "${TASK}" != "addition" && "${TASK}" != "sum_reverse" ]]; then
    echo "unsupported task: ${TASK}" >&2
    exit 2
fi
if [[ "${SUPERVISION}" != "adaptive_step" && "${SUPERVISION}" != "fixed_horizon" ]]; then
    echo "unsupported supervision: ${SUPERVISION}" >&2
    exit 2
fi
if ! [[ "${PHYSICAL_GPU}" =~ ^[0-7]$ ]]; then
    echo "PHYSICAL_GPU must be an integer in [0,7]" >&2
    exit 2
fi
if [[ -z "${BASELINE_VARIANT}" ]]; then
    if [[ "${ACTION}" == official_* ]]; then
        BASELINE_VARIANT=official
    elif [[ "${ACTION}" == released_* ]]; then
        BASELINE_VARIANT=released64
    else
        BASELINE_VARIANT=papertext8
    fi
fi
if [[ "${BASELINE_VARIANT}" != "released64" && "${BASELINE_VARIANT}" != "official" && "${BASELINE_VARIANT}" != "papertext8" ]]; then
    echo "BASELINE_VARIANT must be released64, official, or papertext8" >&2
    exit 2
fi
if [[ "${CONTROLLER_CURRICULUM}" != "extension" && "${CONTROLLER_CURRICULUM}" != "grid" && "${CONTROLLER_CURRICULUM}" != "mixed" && "${CONTROLLER_CURRICULUM}" != "overloop" && "${CONTROLLER_CURRICULUM}" != "logical_range" ]]; then
    echo "unsupported CONTROLLER_CURRICULUM: ${CONTROLLER_CURRICULUM}" >&2
    exit 2
fi
if [[ "${CONTROLLER_CURRICULUM}" == "logical_range" ]]; then
    if ! [[ "${CONTROLLER_LOGICAL_MAX_LENGTH}" =~ ^[0-9]+$ ]] || [[ "${CONTROLLER_LOGICAL_MAX_LENGTH}" -lt 2 ]]; then
        echo "logical_range requires CONTROLLER_LOGICAL_MAX_LENGTH >= 2" >&2
        exit 2
    fi
elif [[ -n "${CONTROLLER_LOGICAL_MAX_LENGTH}" ]]; then
    echo "CONTROLLER_LOGICAL_MAX_LENGTH is only valid for logical_range" >&2
    exit 2
fi
if [[ -n "${CONTROLLER_LOGICAL_MIN_LENGTH}" ]]; then
    if [[ "${CONTROLLER_CURRICULUM}" != "logical_range" ]] || \
       ! [[ "${CONTROLLER_LOGICAL_MIN_LENGTH}" =~ ^[0-9]+$ ]] || \
       [[ "${CONTROLLER_LOGICAL_MIN_LENGTH}" -lt 1 ]] || \
       [[ "${CONTROLLER_LOGICAL_MIN_LENGTH}" -gt "${CONTROLLER_LOGICAL_MAX_LENGTH}" ]]; then
        echo "CONTROLLER_LOGICAL_MIN_LENGTH requires a valid logical_range interval" >&2
        exit 2
    fi
fi
if ! [[ "${CONTROLLER_SEED}" =~ ^[0-9]+$ && "${AUDIT_SEED}" =~ ^[0-9]+$ ]]; then
    echo "CONTROLLER_SEED and AUDIT_SEED must be non-negative integers" >&2
    exit 2
fi
if [[ "${CONTROLLER_FORCE}" != "0" && "${CONTROLLER_FORCE}" != "1" ]]; then
    echo "CONTROLLER_FORCE must be 0 or 1" >&2
    exit 2
fi
if [[ "${CONTROLLER_LR_SCHEDULE}" != "cosine" && "${CONTROLLER_LR_SCHEDULE}" != "wsd" ]]; then
    echo "CONTROLLER_LR_SCHEDULE must be cosine or wsd" >&2
    exit 2
fi
if ! [[ "${CONTROLLER_STABLE_UPDATES}" =~ ^[0-9]+$ ]]; then
    echo "CONTROLLER_STABLE_UPDATES must be a non-negative integer" >&2
    exit 2
fi
if ! [[ "${CONTROLLER_STAGE_ROUND_MULTIPLIER}" =~ ^[1-9][0-9]*$ ]]; then
    echo "CONTROLLER_STAGE_ROUND_MULTIPLIER must be a positive integer" >&2
    exit 2
fi
if [[ "${CONTROLLER_POST_FINAL_J}" != "0" && "${CONTROLLER_POST_FINAL_J}" != "1" ]]; then
    echo "CONTROLLER_POST_FINAL_J must be 0 or 1" >&2
    exit 2
fi
if [[ "${AUDIT_POST_FINAL_J}" != "0" && "${AUDIT_POST_FINAL_J}" != "1" ]]; then
    echo "AUDIT_POST_FINAL_J must be 0 or 1" >&2
    exit 2
fi
if [[ -n "${CONTROLLER_ANCHOR_STEP}" ]] && ! [[ "${CONTROLLER_ANCHOR_STEP}" =~ ^[1-9][0-9]*$ ]]; then
    echo "CONTROLLER_ANCHOR_STEP must be a positive integer" >&2
    exit 2
fi

CODE_DIR=/data/wujiaju/LooPlus
PYTHON_BIN=/data/wujiaju/.venvs/loopreasoner/bin/python
RUN_ROOT=/data/wujiaju/paper_length_telomere_20260731
LOG_ROOT=/data/wujiaju/logs/paper_length_telomere_20260731
if [[ "${BASELINE_VARIANT}" == released64 ]]; then
    LABEL="${TASK}_${SUPERVISION}_released64_seed${SEED}"
elif [[ "${BASELINE_VARIANT}" == official ]]; then
    LABEL="${TASK}_${SUPERVISION}_official_seed${SEED}"
else
    LABEL="${TASK}_${SUPERVISION}_seed${SEED}"
fi
BACKBONE_DIR="${RUN_ROOT}/backbones/${LABEL}"
DIAGNOSIS_DIR="${RUN_ROOT}/diagnosis/${LABEL}"
if [[ "${BASELINE_VARIANT}" == released64 || "${BASELINE_VARIANT}" == official ]]; then
    BACKBONE_CHECKPOINT="${BACKBONE_DIR}/final.pt"
else
    BACKBONE_CHECKPOINT="${BACKBONE_DIR}/best.pt"
fi
if [[ "${CONTROLLER_CURRICULUM}" == "logical_range" ]]; then
    CONTROLLER_LABEL="${LABEL}_rank48_logical1to${CONTROLLER_LOGICAL_MAX_LENGTH}_seed${CONTROLLER_SEED}"
else
    CONTROLLER_LABEL="${LABEL}_rank48_${CONTROLLER_CURRICULUM}_seed${CONTROLLER_SEED}"
fi
if [[ -n "${CONTROLLER_LABEL_OVERRIDE}" ]]; then
    CONTROLLER_LABEL="${CONTROLLER_LABEL_OVERRIDE}"
fi
CONTROLLER_DIR="${RUN_ROOT}/controllers/${CONTROLLER_LABEL}"
AUDIT_DIR="${RUN_ROOT}/audits/${CONTROLLER_LABEL}"
if [[ -n "${AUDIT_DIR_OVERRIDE}" ]]; then
    AUDIT_DIR="${RUN_ROOT}/audits/${AUDIT_DIR_OVERRIDE}"
fi
mkdir -p "${BACKBONE_DIR}" "${DIAGNOSIS_DIR}" "${CONTROLLER_DIR}" "${AUDIT_DIR}" "${LOG_ROOT}"

RESUME_CHECKPOINT=""
RESUME_ARGS=()
if [[ "${ACTION}" == "pilot" || "${ACTION}" == "formal" || "${ACTION}" == "released_formal" || "${ACTION}" == "official_formal" ]]; then
    RESUME_CHECKPOINT="$(find "${BACKBONE_DIR}" -maxdepth 1 -type f -name 'checkpoint_*.pt' | sort | tail -n 1)"
    if [[ -n "${RESUME_CHECKPOINT}" ]]; then
        RESUME_ARGS=(--resume "${RESUME_CHECKPOINT}")
    fi
fi

PRELAUNCH_USED_MIB="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
PRELAUNCH_FREE_MIB="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
PRELAUNCH_UTILIZATION="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"
ACTIVE_PIDS="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true)"
ACTIVE_PIDS_ONE_LINE="$(printf '%s' "${ACTIVE_PIDS}" | tr '\n' ' ' | sed 's/[[:space:]]*$//')"
DECLARED_PEAK_MIB="${DECLARED_PEAK_MIB:-0}"
RESERVE_MIB="${RESERVE_MIB:-16384}"
ALLOW_SHARED_GPU="${ALLOW_SHARED_GPU:-0}"
SHARED_GPU=false
if [[ -n "${ACTIVE_PIDS}" || "${PRELAUNCH_USED_MIB}" -gt 128 ]]; then
    if [[ "${ALLOW_SHARED_GPU}" != "1" ]]; then
        echo "refusing non-empty GPU ${PHYSICAL_GPU}: used=${PRELAUNCH_USED_MIB}MiB pids=${ACTIVE_PIDS:-none}" >&2
        exit 75
    fi
    if ! [[ "${DECLARED_PEAK_MIB}" =~ ^[0-9]+$ ]] || [[ "${DECLARED_PEAK_MIB}" -lt 1 ]]; then
        echo "shared launch requires positive DECLARED_PEAK_MIB" >&2
        exit 2
    fi
    REQUIRED_FREE_MIB=$((DECLARED_PEAK_MIB + RESERVE_MIB))
    if [[ "${PRELAUNCH_FREE_MIB}" -lt "${REQUIRED_FREE_MIB}" ]]; then
        echo "refusing shared GPU ${PHYSICAL_GPU}: free=${PRELAUNCH_FREE_MIB}MiB required=${REQUIRED_FREE_MIB}MiB" >&2
        exit 75
    fi
    SHARED_GPU=true
fi
export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export PYTHONPATH="${CODE_DIR}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${CODE_DIR}"

case "${ACTION}" in
    smoke)
        COMMAND=("${PYTHON_BIN}" -m reasoning_loop.paper_length_telomere backbone
            --task "${TASK}" --supervision "${SUPERVISION}" --steps 20
            --batch-size 8 --d-model 32 --n-heads 4 --d-mlp 64
            --train-max-length 4 --no-amp --log-every 5 --checkpoint-every 20
            --device cuda --seed "${SEED}" --out-dir "${BACKBONE_DIR}_smoke")
        ;;
    benchmark)
        COMMAND=("${PYTHON_BIN}" -m reasoning_loop.paper_length_telomere backbone
            --task "${TASK}" --supervision "${SUPERVISION}" --steps 200
            --batch-size 64 --curriculum-interval 1 --device cuda --seed "${SEED}"
            --log-every 20 --checkpoint-every 200
            --out-dir "${BACKBONE_DIR}_benchmark")
        ;;
    released_benchmark)
        COMMAND=("${PYTHON_BIN}" -m reasoning_loop.paper_length_telomere backbone
            --task "${TASK}" --supervision "${SUPERVISION}" --steps 200
            --batch-size 64 --curriculum-interval 1 --n-heads 64
            --no-amp --device cuda --seed "${SEED}" --log-every 20
            --checkpoint-every 200
            --out-dir "${BACKBONE_DIR}_benchmark")
        ;;
    released_benchmark_fp32)
        COMMAND=("${PYTHON_BIN}" -m reasoning_loop.paper_length_telomere backbone
            --task "${TASK}" --supervision "${SUPERVISION}" --steps 200
            --batch-size 64 --curriculum-interval 1 --n-heads 64
            --no-amp --device cuda --seed "${SEED}" --log-every 20
            --checkpoint-every 200
            --out-dir "${BACKBONE_DIR}_benchmark_fp32")
        ;;
    official_benchmark_fp32)
        COMMAND=("${PYTHON_BIN}" -m reasoning_loop.paper_length_telomere backbone
            --task "${TASK}" --supervision "${SUPERVISION}" --steps 200
            --batch-size 64 --curriculum-interval 1 --official-model-config
            --no-amp --device cuda --seed "${SEED}" --log-every 20
            --checkpoint-every 200
            --out-dir "${BACKBONE_DIR}_benchmark_fp32")
        ;;
    pilot)
        COMMAND=("${PYTHON_BIN}" -m reasoning_loop.paper_length_telomere backbone
            --task "${TASK}" --supervision "${SUPERVISION}" --steps 30000
            --schedule-total-steps 100001
            --device cuda --seed "${SEED}" --log-every 100
            --checkpoint-every 10000 "${RESUME_ARGS[@]}"
            --out-dir "${BACKBONE_DIR}")
        ;;
    formal)
        COMMAND=("${PYTHON_BIN}" -m reasoning_loop.paper_length_telomere backbone
            --task "${TASK}" --supervision "${SUPERVISION}" --steps 100001
            --device cuda --seed "${SEED}" --log-every 100
            --checkpoint-every 10000 "${RESUME_ARGS[@]}"
            --out-dir "${BACKBONE_DIR}")
        ;;
    released_formal)
        COMMAND=("${PYTHON_BIN}" -m reasoning_loop.paper_length_telomere backbone
            --task "${TASK}" --supervision "${SUPERVISION}" --steps 100001
            --n-heads 64 --no-amp --device cuda --seed "${SEED}" --log-every 100
            --checkpoint-every 10000 "${RESUME_ARGS[@]}"
            --out-dir "${BACKBONE_DIR}")
        ;;
    official_formal)
        COMMAND=("${PYTHON_BIN}" -m reasoning_loop.paper_length_telomere backbone
            --task "${TASK}" --supervision "${SUPERVISION}" --steps 100001
            --official-model-config --no-amp --device cuda --seed "${SEED}"
            --log-every 100 --checkpoint-every 10000 "${RESUME_ARGS[@]}"
            --out-dir "${BACKBONE_DIR}")
        ;;
    diagnose)
        read -r -a DIAGNOSIS_LENGTH_ARGS <<< "${DIAGNOSIS_LENGTHS}"
        COMMAND=("${PYTHON_BIN}" -m reasoning_loop.paper_length_telomere diagnose
            --checkpoint "${BACKBONE_CHECKPOINT}"
            --batch-size "${DIAGNOSIS_BATCH_SIZE}" --batches "${DIAGNOSIS_BATCHES}"
            --lengths "${DIAGNOSIS_LENGTH_ARGS[@]}"
            --maximum-step "${DIAGNOSIS_MAXIMUM_STEP}"
            --extension-gate-length "${DIAGNOSIS_EXTENSION_GATE_LENGTH}"
            --device cuda --out-dir "${DIAGNOSIS_DIR}")
        ;;
    controller)
        COMMAND=("${PYTHON_BIN}" -m reasoning_loop.paper_length_telomere controller
            --checkpoint "${BACKBONE_CHECKPOINT}"
            --diagnosis "${DIAGNOSIS_DIR}/summary.json" --rank 48
            --seed "${CONTROLLER_SEED}"
            --controller-curriculum "${CONTROLLER_CURRICULUM}"
            --controller-training-profile "${CONTROLLER_TRAINING_PROFILE}"
            --controller-initialization "${CONTROLLER_INITIALIZATION}"
            --controller-warmup-updates "${CONTROLLER_WARMUP_UPDATES}"
            --controller-final-lr-ratio "${CONTROLLER_FINAL_LR_RATIO}"
            --controller-lr-schedule "${CONTROLLER_LR_SCHEDULE}"
            --controller-stable-updates "${CONTROLLER_STABLE_UPDATES}"
            --controller-ce-temperature "${CONTROLLER_CE_TEMPERATURE}"
            --stage-round-multiplier "${CONTROLLER_STAGE_ROUND_MULTIPLIER}"
            --dense-stage-count "${CONTROLLER_DENSE_STAGE_COUNT}"
            --grad-clip "${CONTROLLER_GRAD_CLIP}"
            --learning-rate-multiplier "${CONTROLLER_LEARNING_RATE_MULTIPLIER}"
            --diagonal-lr-multiplier "${CONTROLLER_DIAGONAL_LR_MULTIPLIER}"
            --device cuda --out-dir "${CONTROLLER_DIR}")
        if [[ "${CONTROLLER_CURRICULUM}" == "logical_range" ]]; then
            COMMAND+=(--controller-logical-max-length "${CONTROLLER_LOGICAL_MAX_LENGTH}")
            if [[ -n "${CONTROLLER_LOGICAL_MIN_LENGTH}" ]]; then
                COMMAND+=(--controller-logical-min-length "${CONTROLLER_LOGICAL_MIN_LENGTH}")
            fi
        fi
        if [[ -n "${CONTROLLER_ANCHOR_STEP}" ]]; then
            COMMAND+=(--controller-anchor-step "${CONTROLLER_ANCHOR_STEP}")
        fi
        if [[ "${CONTROLLER_FORCE}" == "1" ]]; then
            COMMAND+=(--force)
        fi
        if [[ "${CONTROLLER_POST_FINAL_J}" == "1" ]]; then
            COMMAND+=(--controller-post-final-j)
        fi
        ;;
    audit)
        read -r -a AUDIT_LENGTH_ARGS <<< "${AUDIT_LENGTHS}"
        read -r -a AUDIT_MODE_ARGS <<< "${AUDIT_MODES}"
        COMMAND=("${PYTHON_BIN}" -m reasoning_loop.paper_length_telomere audit
            --checkpoint "${BACKBONE_CHECKPOINT}"
            --controller "${CONTROLLER_DIR}/controller.pt"
            --batch-size "${AUDIT_BATCH_SIZE}" --batches "${AUDIT_BATCHES}"
            --lengths "${AUDIT_LENGTH_ARGS[@]}"
            --maximum-step "${AUDIT_MAXIMUM_STEP}"
            --modes "${AUDIT_MODE_ARGS[@]}"
            --seed "${AUDIT_SEED}"
            --device cuda --out-dir "${AUDIT_DIR}")
        if [[ "${AUDIT_POST_FINAL_J}" == "1" ]]; then
            COMMAND+=(--post-final-j)
        fi
        ;;
esac

if [[ "${ACTION}" == "controller" || "${ACTION}" == "audit" ]]; then
    LOG_LABEL="${CONTROLLER_LABEL}"
    MANIFEST_DIR="${RUN_ROOT}/manifests/${CONTROLLER_LABEL}"
    if [[ "${ACTION}" == "audit" && -n "${AUDIT_DIR_OVERRIDE}" ]]; then
        LOG_LABEL="${AUDIT_DIR_OVERRIDE}"
        MANIFEST_DIR="${RUN_ROOT}/manifests/${AUDIT_DIR_OVERRIDE}"
    fi
else
    LOG_LABEL="${LABEL}"
    MANIFEST_DIR="${RUN_ROOT}/manifests/${LABEL}"
fi
RUN_LOG="${LOG_ROOT}/${LOG_LABEL}_${ACTION}.log"
HEARTBEAT_LOG="${LOG_ROOT}/${LOG_LABEL}_${ACTION}.heartbeat.log"
MANIFEST="${MANIFEST_DIR}/${ACTION}.json"
mkdir -p "${MANIFEST_DIR}"
printf '{\n  "status": "launching",\n  "action": "%s",\n  "task": "%s",\n  "supervision": "%s",\n  "baseline_variant": "%s",\n  "seed": %s,\n  "controller_curriculum": "%s",\n  "controller_training_profile": "%s",\n  "controller_initialization": "%s",\n  "controller_force": %s,\n  "controller_anchor_step": "%s",\n  "controller_seed": %s,\n  "controller_lr_schedule": "%s",\n  "controller_stable_updates": %s,\n  "controller_post_final_j": %s,\n  "audit_post_final_j": %s,\n  "audit_seed": %s,\n  "resume_checkpoint": "%s",\n  "physical_gpu": %s,\n  "shared_gpu": %s,\n  "prelaunch_used_mib": %s,\n  "prelaunch_free_mib": %s,\n  "prelaunch_utilization_percent": %s,\n  "preexisting_pids": "%s",\n  "declared_peak_mib": %s,\n  "reserve_mib": %s,\n  "log": "%s"\n}\n' \
    "${ACTION}" "${TASK}" "${SUPERVISION}" "${BASELINE_VARIANT}" \
    "${SEED}" "${CONTROLLER_CURRICULUM}" \
    "${CONTROLLER_TRAINING_PROFILE}" "${CONTROLLER_INITIALIZATION}" \
    "${CONTROLLER_FORCE}" \
    "${CONTROLLER_ANCHOR_STEP}" "${CONTROLLER_SEED}" \
    "${CONTROLLER_LR_SCHEDULE}" "${CONTROLLER_STABLE_UPDATES}" \
    "${CONTROLLER_POST_FINAL_J}" "${AUDIT_POST_FINAL_J}" \
    "${AUDIT_SEED}" "${RESUME_CHECKPOINT}" "${PHYSICAL_GPU}" \
    "${SHARED_GPU}" "${PRELAUNCH_USED_MIB}" "${PRELAUNCH_FREE_MIB}" \
    "${PRELAUNCH_UTILIZATION}" "${ACTIVE_PIDS_ONE_LINE}" \
    "${DECLARED_PEAK_MIB}" "${RESERVE_MIB}" "${RUN_LOG}" > "${MANIFEST}"

"${COMMAND[@]}" > "${RUN_LOG}" 2>&1 &
RUN_PID=$!
printf '%s pid=%s action=%s task=%s supervision=%s gpu=%s status=running\n' \
    "$(date -Is)" "${RUN_PID}" "${ACTION}" "${TASK}" "${SUPERVISION}" "${PHYSICAL_GPU}" >> "${HEARTBEAT_LOG}"
while kill -0 "${RUN_PID}" 2>/dev/null; do
    CURRENT_FREE_MIB="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    RUNTIME_STATUS=running
    if [[ "${CURRENT_FREE_MIB}" -lt "${RESERVE_MIB}" ]]; then
        if [[ "${SHARED_GPU}" == true ]]; then
            RUNTIME_STATUS=retreat_below_reserve
            printf '%s pid=%s action=%s task=%s supervision=%s gpu=%s status=%s free_mib=%s\n' \
                "$(date -Is)" "${RUN_PID}" "${ACTION}" "${TASK}" "${SUPERVISION}" "${PHYSICAL_GPU}" "${RUNTIME_STATUS}" "${CURRENT_FREE_MIB}" >> "${HEARTBEAT_LOG}"
            kill -TERM "${RUN_PID}" 2>/dev/null || true
            break
        fi
        RUNTIME_STATUS=running_below_reserve_exclusive
    fi
    CURRENT_OTHER_PIDS="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' | grep -vx "${RUN_PID}" || true)"
    while IFS= read -r CURRENT_PID; do
        [[ -z "${CURRENT_PID}" ]] && continue
        if ! printf '%s\n' "${ACTIVE_PIDS}" | grep -qx "${CURRENT_PID}"; then
            RUNTIME_STATUS="running_new_colocated_pid_no_retreat:${CURRENT_PID}"
        fi
    done <<< "${CURRENT_OTHER_PIDS}"
    printf '%s pid=%s action=%s task=%s supervision=%s gpu=%s status=%s free_mib=%s\n' \
        "$(date -Is)" "${RUN_PID}" "${ACTION}" "${TASK}" "${SUPERVISION}" "${PHYSICAL_GPU}" "${RUNTIME_STATUS}" "${CURRENT_FREE_MIB}" >> "${HEARTBEAT_LOG}"
    sleep 30
done
set +e
wait "${RUN_PID}"
STATUS=$?
set -e
if [[ "${STATUS}" -eq 0 ]]; then RUN_STATUS=complete; else RUN_STATUS=failed; fi
printf '%s pid=%s action=%s task=%s supervision=%s gpu=%s status=%s exit=%s\n' \
    "$(date -Is)" "${RUN_PID}" "${ACTION}" "${TASK}" "${SUPERVISION}" "${PHYSICAL_GPU}" "${RUN_STATUS}" "${STATUS}" >> "${HEARTBEAT_LOG}"
printf '{\n  "status": "%s",\n  "exit_code": %s,\n  "pid": %s,\n  "action": "%s",\n  "task": "%s",\n  "supervision": "%s",\n  "baseline_variant": "%s",\n  "seed": %s,\n  "controller_curriculum": "%s",\n  "controller_training_profile": "%s",\n  "controller_initialization": "%s",\n  "controller_force": %s,\n  "controller_anchor_step": "%s",\n  "controller_seed": %s,\n  "controller_lr_schedule": "%s",\n  "controller_stable_updates": %s,\n  "controller_post_final_j": %s,\n  "audit_post_final_j": %s,\n  "audit_seed": %s,\n  "resume_checkpoint": "%s",\n  "physical_gpu": %s,\n  "shared_gpu": %s,\n  "prelaunch_used_mib": %s,\n  "prelaunch_free_mib": %s,\n  "prelaunch_utilization_percent": %s,\n  "preexisting_pids": "%s",\n  "declared_peak_mib": %s,\n  "reserve_mib": %s,\n  "log": "%s"\n}\n' \
    "${RUN_STATUS}" "${STATUS}" "${RUN_PID}" "${ACTION}" "${TASK}" \
    "${SUPERVISION}" "${BASELINE_VARIANT}" "${SEED}" \
    "${CONTROLLER_CURRICULUM}" "${CONTROLLER_TRAINING_PROFILE}" \
    "${CONTROLLER_INITIALIZATION}" "${CONTROLLER_FORCE}" \
    "${CONTROLLER_ANCHOR_STEP}" \
    "${CONTROLLER_SEED}" "${CONTROLLER_LR_SCHEDULE}" \
    "${CONTROLLER_STABLE_UPDATES}" "${CONTROLLER_POST_FINAL_J}" \
    "${AUDIT_POST_FINAL_J}" "${AUDIT_SEED}" \
    "${RESUME_CHECKPOINT}" \
    "${PHYSICAL_GPU}" "${SHARED_GPU}" \
    "${PRELAUNCH_USED_MIB}" "${PRELAUNCH_FREE_MIB}" \
    "${PRELAUNCH_UTILIZATION}" "${ACTIVE_PIDS_ONE_LINE}" \
    "${DECLARED_PEAK_MIB}" "${RESERVE_MIB}" "${RUN_LOG}" > "${MANIFEST}"
exit "${STATUS}"
