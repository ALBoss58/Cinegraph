#!/usr/bin/env bash
# Cinegraph installer for macOS / Linux
set -e
if ! command -v ffmpeg >/dev/null; then
  if command -v brew >/dev/null; then brew install ffmpeg
  elif command -v apt-get >/dev/null; then sudo apt-get update && sudo apt-get install -y ffmpeg
  else echo "Install ffmpeg first (https://ffmpeg.org/download.html)"; exit 1; fi
fi
python3 -m venv "${HOME}/.cinegraph/venv"
"${HOME}/.cinegraph/venv/bin/pip" install --upgrade pip
"${HOME}/.cinegraph/venv/bin/pip" install -e "$(dirname "$0")/..[ai,full]"
echo
echo "Done. Add to PATH:  export PATH=\"$HOME/.cinegraph/venv/bin:\$PATH\""
echo "Try:  cinegraph info"
