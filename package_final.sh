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

STAGE=$(mktemp -d)
mkdir -p "$STAGE/code" "$STAGE/report"
cp -r code/* "$STAGE/code/"
cp "$PDF" "$STAGE/report/"
find "$STAGE" -name '.ipynb_checkpoints' -prune -exec rm -rf {} +
find "$STAGE" -name '__pycache__' -prune -exec rm -rf {} +

(cd "$STAGE" && zip -qr "$OLDPWD/${NAME}-report_code.zip" code report)
rm -rf "$STAGE"
echo "Built ${NAME}-report_code.zip"; unzip -l "${NAME}-report_code.zip" | head -20
