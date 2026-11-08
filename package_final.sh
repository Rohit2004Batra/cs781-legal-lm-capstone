#!/usr/bin/env bash
# Builds the Hello IITK deliverable:
#   CS781-CP-<First>-<Last>-<RollNo>-report_code.zip
#   ├── code/phase1/ ... code/phase7/
#   └── report/CS781-CP-<First>-<Last>-<RollNo>-report.pdf
# with code/ and report/ directly at the zip root (no wrapper folder).
#
# Usage: ./package_final.sh Firstname Lastname RollNo
set -euo pipefail
FIRST=$1; LAST=$2; ROLL=$3
NAME="CS781-CP-${FIRST}-${LAST}-${ROLL}"
PDF="report/${NAME}-report.pdf"
[ -f "$PDF" ] || { echo "Missing $PDF — compile the report and name it exactly that."; exit 1; }
if command -v pdftotext >/dev/null && pdftotext "$PDF" - | grep -c -e "{userid}" -e "Write 8--12 sentences" -e "Write 8–12 sentences" >/dev/null; then
  echo "WARNING: the PDF still contains placeholders (user ID or Q4). Fill them in Report.tex and recompile."; fi

STAGE=$(mktemp -d)
mkdir -p "$STAGE/code" "$STAGE/report"
cp -r code/* "$STAGE/code/"
cp README.md "$STAGE/code/README.md"                 # repository overview + how to run
mkdir -p "$STAGE/code/phase1/tokenizer" && cp tokenizer/* "$STAGE/code/phase1/tokenizer/"   # tokenizer config (not weights)
cp "$PDF" "$STAGE/report/"
find "$STAGE" -name '.ipynb_checkpoints' -prune -exec rm -rf {} +
find "$STAGE" -name '__pycache__' -prune -exec rm -rf {} +

BAD=$(find "$STAGE" -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.bin' -o -name '*.ckpt' -o -name '*.safetensors' \))
[ -z "$BAD" ] || { echo "Checkpoint files found, not allowed in the zip:"; echo "$BAD"; exit 1; }
rm -f "${NAME}-report_code.zip"
(cd "$STAGE" && zip -qr "$OLDPWD/${NAME}-report_code.zip" code report)
rm -rf "$STAGE"
echo "Built ${NAME}-report_code.zip"; unzip -l "${NAME}-report_code.zip" | head -20
