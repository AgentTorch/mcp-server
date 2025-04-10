#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}AgentTorch MCP Server starting...${NC}"

# Create necessary directories if they don't exist
mkdir -p /app/static
mkdir -p /app/templates
mkdir -p /app/data/18x25
mkdir -p /app/services

# Check if the server mode is provided via environment variable or command line arg
SERVER_MODE=${1:-${SERVER_MODE:-full}}

echo -e "${GREEN}Running in $SERVER_MODE mode${NC}"

# Check for API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo -e "${YELLOW}Warning: ANTHROPIC_API_KEY is not set. Using mock LLM responses.${NC}"
else
    echo -e "${GREEN}Using Anthropic API key from environment${NC}"
fi

# Run the server based on mode
case "$SERVER_MODE" in
    full)
        echo -e "${GREEN}Running all services: MCP, WebSocket, and Web UI${NC}"
        python server.py
        ;;
    web)
        echo -e "${GREEN}Running Web UI and WebSocket server${NC}"
        python server.py web
        ;;
    mcp)
        echo -e "${GREEN}Running MCP server only (for Claude Desktop)${NC}"
        python server.py run
        ;;
    websocket)
        echo -e "${GREEN}Running WebSocket server only${NC}"
        python server.py websocket
        ;;
    *)
        echo -e "${YELLOW}Unknown mode: $SERVER_MODE, defaulting to full${NC}"
        python server.py
        ;;
esac