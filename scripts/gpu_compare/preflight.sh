#!/bin/bash
set -euo pipefail

GPU_LABEL="${GPU_LABEL:-5090}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${REQUIRE_GPU:-0}" == "1" ]]; then
    bash "${SCRIPT_DIR}/preflight_gpu.sh"
else
    bash "${SCRIPT_DIR}/preflight_no_card.sh"
fi
