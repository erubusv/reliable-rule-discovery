#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DESTINATION="${1:-${ROOT}/artifacts/anonymous-release}"

if [[ -e "${DESTINATION}" ]]; then
    echo "destination already exists: ${DESTINATION}" >&2
    exit 1
fi

mkdir -p "${DESTINATION}"
cd "${ROOT}"
while IFS= read -r -d '' path; do
    if [[ -e "${path}" || -L "${path}" ]]; then
        printf '%s\0' "${path}"
    fi
done < <(git ls-files --cached --others --exclude-standard -z) \
    | tar --null --files-from=- --create \
    | tar --extract --directory="${DESTINATION}"

git -C "${DESTINATION}" init >/dev/null
git -C "${DESTINATION}" checkout -b main >/dev/null 2>&1
git -C "${DESTINATION}" config user.name "Anonymous Authors"
git -C "${DESTINATION}" config user.email "anonymous@users.noreply.github.com"
git -C "${DESTINATION}" add --all
GIT_AUTHOR_NAME="Anonymous Authors" \
GIT_AUTHOR_EMAIL="anonymous@users.noreply.github.com" \
GIT_AUTHOR_DATE="2000-01-01T00:00:00Z" \
GIT_COMMITTER_NAME="Anonymous Authors" \
GIT_COMMITTER_EMAIL="anonymous@users.noreply.github.com" \
GIT_COMMITTER_DATE="2000-01-01T00:00:00Z" \
    git -C "${DESTINATION}" commit --message "Initial anonymous release" >/dev/null

echo "anonymous release created at ${DESTINATION}"
echo "push this repository to a new anonymous remote; do not reuse development history"
