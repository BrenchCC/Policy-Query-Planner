#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"
mode="${1:-test}"

cd "${project_root}"

workers="${WORKERS:-4}"
max_retries="${MAX_RETRIES:-2}"
log_every="${LOG_EVERY:-10}"
test_limit="${TEST_LIMIT:-2}"
test_workers="${TEST_WORKERS:-2}"

run_stage() {
    local stage="$1"

    if [[ "${mode}" == "test" ]]; then
        echo "Testing ${stage}: limit=${test_limit}, workers=${test_workers}"
        python data_preprocess/generate_with_ark.py \
            --stage "${stage}" \
            --limit "${test_limit}" \
            --workers "${test_workers}" \
            --max-retries "${max_retries}" \
            --log-every 1 \
            --resume
        return
    fi

    echo "Running ${stage}: workers=${workers}, max_retries=${max_retries}"
    python data_preprocess/generate_with_ark.py \
        --stage "${stage}" \
        --workers "${workers}" \
        --max-retries "${max_retries}" \
        --log-every "${log_every}" \
        --resume
}

if [[ "${mode}" != "test" && "${mode}" != "full" ]]; then
    echo "Usage: $0 [test|full]" >&2
    exit 2
fi

for stage in sft dpo grpo; do
    run_stage "${stage}"
done

if [[ "${mode}" == "full" ]]; then
    echo "Finalizing API-enhanced datasets"
    python data_preprocess/finalize_datasets.py --stage all

    echo "Validating all datasets"
    python data_preprocess/validate_datasets.py --stage all
fi

echo "API generation mode '${mode}' completed"
