#!/usr/bin/env bash
# Regenerate Fed-CORE_draft.docx from the markdown source.
# Workflow: Fed-CORE_draft.md is the editable source of truth; run this after any
# edit to refresh the Word file (figures + LaTeX math + tables are carried over).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

SRC="docs/Fed-CORE_draft.md"
OUT="docs/Fed-CORE_draft.docx"

# Information Sciences (Elsevier) manuscript style: no TOC (post-processed for
# Times New Roman 12pt, double spacing, continuous line numbers, page numbers).
pandoc "$SRC" -o "$OUT" \
  --resource-path=.:experiments/fedcore/figs \
  -V geometry:margin=1in

python3 ins_format.py "$OUT"

echo "[build_docx] wrote $OUT ($(du -h "$OUT" | cut -f1))"
