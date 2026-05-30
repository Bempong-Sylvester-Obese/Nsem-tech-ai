#!/usr/bin/env bash
# Remove .venv/ from all git history (run once after accidental commit).
# WARNING: Rewrites history. All collaborators must re-clone or reset hard.
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v git-filter-repo >/dev/null 2>&1; then
  echo "Install: pip install git-filter-repo"
  exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Commit or stash changes before rewriting history."
  exit 1
fi

REMOTE="${1:-origin}"
REMOTE_URL="$(git remote get-url "$REMOTE" 2>/dev/null || true)"

echo "Removing .venv/ from entire history..."
git filter-repo --path .venv --invert-paths --force

if [[ -n "$REMOTE_URL" ]]; then
  git remote add "$REMOTE" "$REMOTE_URL" 2>/dev/null || git remote set-url "$REMOTE" "$REMOTE_URL"
fi

git reflog expire --expire=now --all
git gc --prune=now

echo ""
echo "Done. Verify: git rev-list --objects --all | grep -c '\\.venv/'  (expect 0)"
echo "Then force-push:"
echo "  git push --force $REMOTE main"
echo "  git push --force $REMOTE --all"
echo ""
echo "Tell collaborators to re-clone the repository."
