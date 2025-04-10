#!/bin/bash
# Start the websocket server in the background
python server.py websocket &
WS_PID=$!

# Start the MCP server
mcp run server.py

# If MCP server exits, kill the websocket server
kill $WS_PID