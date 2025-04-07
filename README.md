# Running the AgentTorch MCP Server

This guide explains how to run the AgentTorch MCP server both locally and with Claude Desktop integration using Docker.

## Option 1: Running Locally with Docker

### Prerequisites
- Docker installed on your machine
- Docker Compose installed on your machine
- Git installed on your machine

### Step 1: Clone the Repository
```bash
git clone https://github.com/AgentTorch/mcp-server
cd mcp-server
```

### Step 2: Set Up Environment Files
Create an `.env` file in the project root with your Anthropic API key:
```bash
# Create .env file
cat > .env << 'EOL'
ANTHROPIC_API_KEY=api_key


PYTHONPATH=/app
AGENTTORCH_CONFIG_PATH=/app/config.yaml
EOL
```

### Step 3: Launch with Docker Compose
```bash
# Start the containers
docker-compose up -d

# Check if the containers are running
docker-compose ps

# View logs
docker-compose logs -f
```

### Step 4: Access the Web UI
Open your browser and go to:
```
http://localhost:8080
```

### Step 5: Run the MCP Server Locally
You can interact with the containerized MCP server using the following command:
```bash
# Execute the server directly in the container
docker exec -it agenttorch-mcp mcp run server.py
```

## Option 2: Claude Desktop Integration

### Step 1: Create Required Files
Ensure you've completed Steps 1-3 from Option 1 above to get the Docker container running.

### Step 2: Configure Claude Desktop
Create or update Claude Desktop configuration:

1. Locate your Claude Desktop configuration directory:
   - macOS: `~/.claude-desktop/`
   - Windows: `%APPDATA%\claude-desktop\`
   - Linux: `~/.config/claude-desktop/`

2. Create or edit the config.json file:
```bash
# For macOS/Linux
mkdir -p ~/.claude-desktop
cat > ~/.claude-desktop/config.json << 'EOL'
{
  "mcpServers": {
    "AgentTorch Simulator": {
      "command": "docker",
      "args": [
        "exec",
        "-it",
        "agenttorch-mcp",
        "mcp",
        "run",
        "server.py"
      ]
    }
  }
}
EOL

# For Windows (using PowerShell)
# New-Item -Path "$env:APPDATA\claude-desktop" -ItemType Directory -Force
# Set-Content -Path "$env:APPDATA\claude-desktop\config.json" -Value '{
#   "mcpServers": {
#     "AgentTorch Simulator": {
#       "command": "docker",
#       "args": [
#         "exec",
#         "-it",
#         "agenttorch-mcp",
#         "mcp",
#         "run",
#         "server.py"
#       ]
#     }
#   }
# }'
```

### Step 3: Launch Claude Desktop
1. Open Claude Desktop
2. You should see "AgentTorch Simulator" in the server selection dropdown
3. Select it to connect to your Docker container

### Step 4: Test Integration
1. Start a new conversation in Claude Desktop
2. Select one of the available prompts (e.g., "Run Simulation")
3. Enter a query like "Run a simulation with 500 predators and 1000 prey"
4. Check both Claude Desktop and the web UI (http://localhost:8080) for results

## Troubleshooting

### Docker Issues
```bash
# Restart the container
docker-compose restart

# Rebuild the container (if you made changes)
docker-compose down
docker-compose up -d --build

# Check container logs
docker-compose logs -f
```

### Claude Desktop Connection Issues
1. Verify Docker container is running:
   ```bash
   docker ps | grep agenttorch-mcp
   ```

2. Check if the container can execute MCP commands:
   ```bash
   docker exec -it agenttorch-mcp mcp list
   ```

3. Restart Claude Desktop

### Web UI Issues
1. Check if the HTTP server is running:
   ```bash
   docker exec -it agenttorch-mcp curl http://localhost:8080
   ```

2. Check port mappings:
   ```bash
   docker-compose ps
   ```

## Container Management Commands

```bash
# Stop the container
docker-compose stop

# Start the container
docker-compose start

# Remove the container (keeps your data)
docker-compose down

# Remove the container and volumes (deletes your data)
docker-compose down -v

# View container resource usage
docker stats
```