#!/usr/bin/env bash
# Launch the OpenROAD Flow web UI.
# Open http://localhost:5000 in a browser after running this.
cd "$(dirname "$0")"
python3 ui/app.py
