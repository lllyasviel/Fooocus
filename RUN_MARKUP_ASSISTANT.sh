#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
echo "Starting Fooocus Local Markup Assistant at http://127.0.0.1:7871"
python local_markup_app.py
