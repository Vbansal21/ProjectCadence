#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <YYYYWww> <slug>"
  echo "Example: $0 2025W33 vortex-viz"
  exit 1
fi

WEEK_ID="$1"
SLUG="$2"
DEST="weeks/${WEEK_ID}_${SLUG}"

if [[ -e "$DEST" ]]; then
  echo "[new-week] Already exists: $DEST"
  exit 1
fi

mkdir -p "$DEST"/{src,notebooks,tests,data,docs,scripts}

cat > "$DEST/README.md" <<EOF
# ${WEEK_ID} — ${SLUG}

**Goal**: _one-sentence target_

## How to run
\`\`\`bash
# example
python -m ${SLUG//-/_}.main
\`\`\`

## Notes
- key decisions
- what worked / what failed
- next steps
EOF

# keep empty data dir tracked
touch "$DEST/data/.gitkeep"

echo "[new-week] Created $DEST"
