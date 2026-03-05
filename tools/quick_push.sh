#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

git add -A

if git diff --cached --quiet; then
  echo "No changes to commit."
  exit 0
fi

msg="${1:-}"
if [[ -z "$msg" ]]; then
  msg="update: $(date +%Y-%m-%d_%H-%M-%S)"
fi

git commit -m "$msg"
git push
