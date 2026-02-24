#!/bin/bash
# Runs single-example GRPO training sequentially.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Starting GRPO training (single DSR) ==="
"$SCRIPT_DIR/train_single_dsr.sh"

echo "=== Starting GRPO training (single RLVR) ==="
"$SCRIPT_DIR/train_single.sh"

echo "=== All training complete ==="
