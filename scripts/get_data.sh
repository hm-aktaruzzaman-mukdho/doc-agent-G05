#!/usr/bin/env bash
# A1 — fetch or recreate your scanned corpus into data/raw/
# IMPLEMENT: download/prepare your corpus (public-domain / openly licensed).
set -euo pipefail

OUT_DIR="$(dirname "$0")/../data/raw"
mkdir -p "$OUT_DIR"

BOOKS=(
  "history_world_civilization|https://drive.google.com/uc?export=download&id=1fIva2037ZxRmu3kZRgRevoWt18yBo_22"
  "bangladesh_global_studies_civics|https://drive.google.com/uc?export=download&id=1y8nVlBXO8JA7yCVka_kaf2xFQZl5alS2"
  "geography_environment|https://drive.google.com/uc?export=download&id=1IBxZA_tZ8Ayhcm0s7IAOXaju0iFt2Xsc"
)

for entry in "${BOOKS[@]}"; do
  IFS='|' read -r name url <<<"${entry}"
  dest="${OUT_DIR}/${name}.pdf"
  echo "Fetching ${name} from ${url} ..."
  if curl -fSL --retry 3 -o "${dest}" "${url}"; then
    echo "  -> saved to ${dest}"
  else
    echo "  !! FAILED to fetch ${name}. Check the URL is current on nctb.gov.bd and update this script." >&2
  fi
done

echo ""
echo "Done. Verify each PDF opens correctly before running notebooks/eda.ipynb:"
for entry in "${BOOKS[@]}"; do
  IFS='|' read -r name url <<<"${entry}"
  echo "  - ${OUT_DIR}/${name}.pdf"
done
