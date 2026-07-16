#!/bin/bash

set -euo pipefail

# Compatibility wrapper: run the generalized Track3 DDP probe for MuonEq only.
OPTS="${OPTS:-softeq-k2000-muon}" bash scripts/gpu_compare/track3_optimizer_probe.sh "$@"
