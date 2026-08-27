#!/usr/bin/env bash
# Deploy Chakranetra to a Hugging Face Space.
#
#   HF_TOKEN=hf_xxx HF_USER=your-username ./deploy/huggingface/push_space.sh
#
# Builds a clean worktree containing the app plus the Space-specific Dockerfile
# and README front matter, then pushes it to the Space repo. Nothing here
# touches the GitHub remote.
set -euo pipefail

: "${HF_TOKEN:?set HF_TOKEN}"
: "${HF_USER:?set HF_USER}"
SPACE_NAME="${SPACE_NAME:-chakranetra}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

echo "==> staging app into $STAGE"
cd "$REPO_ROOT"
git archive HEAD | tar -x -C "$STAGE"

# Space-specific overrides: Docker SDK metadata and the uid-1000 / port-7860 image.
cp deploy/huggingface/Dockerfile "$STAGE/Dockerfile"
cp deploy/huggingface/README.md  "$STAGE/README.md"
rm -rf "$STAGE/deploy" "$STAGE/.github"

cd "$STAGE"
git init -q -b main
git config user.name  "$HF_USER"
git config user.email "$HF_USER@users.noreply.huggingface.co"
git add -A
git commit -q -m "Deploy Chakranetra"

echo "==> pushing to https://huggingface.co/spaces/$HF_USER/$SPACE_NAME"
git remote add space "https://$HF_USER:$HF_TOKEN@huggingface.co/spaces/$HF_USER/$SPACE_NAME"
git push -q --force space main

echo "==> done: https://huggingface.co/spaces/$HF_USER/$SPACE_NAME"
