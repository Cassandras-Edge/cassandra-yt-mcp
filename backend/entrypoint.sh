#!/bin/sh
set -e

# Start bgutil PO token server in background (required for YouTube downloads)
if [ -f /opt/bgutil-server/build/main.js ]; then
  node /opt/bgutil-server/build/main.js --port 4416 &
  BGUTIL_PID=$!
  echo "bgutil PO token server started (PID $BGUTIL_PID, port 4416)"
fi

# Run the main application
exec "$@"
