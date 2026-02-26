#!/usr/bin/env bash

set -euo pipefail

center_block() {
  local cols=${COLUMNS:-$(tput cols)}
  local -a lines=()
  local max=0

  # Read all lines (preserve spaces and backslashes)
  while IFS= read -r line; do
    lines+=("$line")
    (( ${#line} > max )) && max=${#line}
  done

  # Compute left padding based on the longest line
  local pad=$(( (cols - max) / 2 ))
  (( pad < 0 )) && pad=0

  # Print with uniform padding so the art stays aligned
  for line in "${lines[@]}"; do
    printf "%*s%s\n" "$pad" "" "$line"
  done
}

center_block << 'EOF'
 ██████╗ ██████╗ ███████╗███╗   ██╗ ██████╗██╗  ██╗██╗██████╗
 ██╔═══██╗██╔══██╗██╔════╝████╗  ██║██╔════╝██║  ██║██║██╔══██╗
██║   ██║██████╔╝█████╗  ██╔██╗ ██║██║     ███████║██║██████╔╝
██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║██║     ██╔══██║██║██╔═══╝
╚██████╔╝██║     ███████╗██║ ╚████║╚██████╗██║  ██║██║██║
 ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝ ╚═════╝╚═╝  ╚═╝╚═╝╚═╝                                                                                                                     
EOF

center_block << 'EOF'
                                                                                                                                                                             
███╗   ███╗██╗    ██╗ ██████╗██████╗  ██████╗
████╗ ████║██║    ██║██╔════╝╚════██╗██╔════╝
██╔████╔██║██║ █╗ ██║██║      █████╔╝███████╗
██║╚██╔╝██║██║███╗██║██║     ██╔═══╝ ██╔═══██╗
██║ ╚═╝ ██║╚███╔███╔╝╚██████╗███████╗╚██████╔╝
╚═╝     ╚═╝ ╚══╝╚══╝  ╚═════╝╚══════╝ ╚═════╝
EOF

center_block << 'EOF'
████████╗ █████╗ ██╗     ███████╗███╗   ██╗████████╗     █████╗ ██████╗ ███████╗███╗   ██╗ █████╗         ██╗    ██╗ ██████╗ ██████╗ ██╗  ██╗███████╗██╗  ██╗ ██████╗ ██████╗ 
╚══██╔══╝██╔══██╗██║     ██╔════╝████╗  ██║╚══██╔══╝    ██╔══██╗██╔══██╗██╔════╝████╗  ██║██╔══██╗        ██║    ██║██╔═══██╗██╔══██╗██║ ██╔╝██╔════╝██║  ██║██╔═══██╗██╔══██╗
   ██║   ███████║██║     █████╗  ██╔██╗ ██║   ██║       ███████║██████╔╝█████╗  ██╔██╗ ██║███████║        ██║ █╗ ██║██║   ██║██████╔╝█████╔╝ ███████╗███████║██║   ██║██████╔╝
   ██║   ██╔══██║██║     ██╔══╝  ██║╚██╗██║   ██║       ██╔══██║██╔══██╗██╔══╝  ██║╚██╗██║██╔══██║        ██║███╗██║██║   ██║██╔══██╗██╔═██╗ ╚════██║██╔══██║██║   ██║██╔═══╝ 
   ██║   ██║  ██║███████╗███████╗██║ ╚████║   ██║       ██║  ██║██║  ██║███████╗██║ ╚████║██║  ██║        ╚███╔███╔╝╚██████╔╝██║  ██║██║  ██╗███████║██║  ██║╚██████╔╝██║     
   ╚═╝   ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═══╝   ╚═╝       ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═══╝╚═╝  ╚═╝         ╚══╝╚══╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝     
                                                                                                                                                                              
EOF

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Installing uv..."

  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | sh
  else
    echo "Error: neither curl nor wget is available to install uv." >&2
    exit 1
  fi

  export PATH="$HOME/.local/bin:$PATH"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv is still not available after installation." >&2
  exit 1
fi

echo "Syncing project dependencies with uv..."
uv sync
echo "Environment is ready."
