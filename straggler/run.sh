#!/usr/bin/env bash
# ============================================================
#  PLABS – Full pipeline runner (Ubuntu)
#  Usage:  bash run.sh
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_ROOT="${SCRIPT_DIR}/../AUtool-Dataset"
SRC="${SCRIPT_DIR}/src"
OUT="${SCRIPT_DIR}/outputs"
export AUTOOL_DATASET="${DATASET_ROOT}"
export TF_CPP_MIN_LOG_LEVEL=2

GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
log(){ echo -e "${CYAN}▶ $1${NC}"; }
ok() { echo -e "${GREEN}✔ $1${NC}"; }

# ── 0. Dataset check ──────────────────────────────────────────────────────────
log "Checking AUtool-Dataset…"
if [ ! -d "${DATASET_ROOT}/Heterogeneity" ]; then
  echo "Dataset not found at ${DATASET_ROOT}"
  echo "Cloning from GitHub…"
  git clone https://github.com/Xyl0014742/AUtool-Dataset.git \
      "${DATASET_ROOT}"
fi
ok "Dataset ready"

# ── 1. Python dependencies ───────────────────────────────────────────────────
log "Installing Python dependencies…"
pip install pandas numpy scikit-learn imbalanced-learn tensorflow \
            flask matplotlib seaborn joblib --break-system-packages -q
ok "Dependencies installed"

# ── 2. Create output dirs ─────────────────────────────────────────────────────
mkdir -p "${OUT}/models" "${OUT}/plots"

# ── 3. Run pipeline ───────────────────────────────────────────────────────────
log "Running full PLABS pipeline…"
python "${SRC}/pipeline.py"

ok "Pipeline complete!"
echo ""
echo "  📊 Results:        ${OUT}/"
echo "  📄 HTML report:    ${OUT}/report.html"
echo "  📈 Metrics CSV:    ${OUT}/metrics.csv"
echo "  🎯 Events CSV:     ${OUT}/detection_events.csv"
echo ""
echo "  To open the dashboard:"
echo "    python ${SCRIPT_DIR}/dashboard/app.py"
echo "    Then open  http://localhost:5050  in your browser"
