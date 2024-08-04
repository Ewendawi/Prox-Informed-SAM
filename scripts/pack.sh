#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR_NAME="notebook_pack"
OUT_DIR="${ROOT_DIR}/${OUT_DIR_NAME}"
ZIP_PATH="${ROOT_DIR}/${OUT_DIR_NAME}.zip"
TAR_PATH="${ROOT_DIR}/${OUT_DIR_NAME}.tar.gz"

echo "[pack] Root dir: ${ROOT_DIR}"

echo "[pack] build new lib"

python -m build .

if [[ ! -d "${ROOT_DIR}/dist" ]]; then
  echo "ERROR: dist/ directory not found next to scripts/." >&2
  exit 1
fi

echo "[pack] Cleaning old outputs..."
rm -rf "${OUT_DIR}" "${ZIP_PATH}" "${TAR_PATH}"

echo "[pack] Creating ${OUT_DIR_NAME}/..."
mkdir -p "${OUT_DIR}"

echo "[pack] Copying dist/..."
cp -r "${ROOT_DIR}/dist" "${OUT_DIR}/"
cp -r "${ROOT_DIR}/scripts" "${OUT_DIR}/"

echo "[pack] Creating archive..."
pushd "${ROOT_DIR}" >/dev/null
if command -v zip >/dev/null 2>&1; then
  zip -r "${OUT_DIR_NAME}.zip" "${OUT_DIR_NAME}" >/dev/null
  echo "[pack] Wrote zip archive: ${ZIP_PATH}"
else
  tar czf "${OUT_DIR_NAME}.tar.gz" "${OUT_DIR_NAME}"
  echo "[pack] zip not found, wrote tar.gz archive: ${TAR_PATH}"
fi
popd >/dev/null

echo "[pack] Removing ${OUT_DIR_NAME}/ directory..."
rm -rf "${OUT_DIR}"

echo "[pack] Done."
