#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
echo "Starting Fooocus locally..."
python launch.py --disable-analytics
