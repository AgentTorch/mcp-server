# AgentTorch MCP Server

## Overview

AgentTorch MCP Server provides an interactive predator-prey simulation platform leveraging the Model Context Protocol (MCP) to create engaging ecological simulations. It connects AgentTorch's PyTorch-based simulations with Claude to deliver AI-powered analysis and visualization of complex ecosystem dynamics.

The platform enables users to run simulations, modify parameters through conversation, analyze results with AI insights, and visualize population trends.

## Features

- **Interactive Simulation**: Configure and run ecological simulations with simple commands
- **Substep Customization**: Modify movement, feeding, and hunting mechanics
- **AI-Powered Analysis**: Get detailed ecological analysis of simulation results
- **Web UI Integration**: View population trends and simulation logs visually
- **Claude Desktop Integration**: Interact naturally using conversation

## Prerequisites

- Python 3.11+
- uv (Python package installer)
- Claude Desktop (for MCP integration)

## Installation & Setup

1. **Clone the repository**:
```bash
git clone https://github.com/AgentTorch/mcp-server
cd mcp-server
```

2. **Install MCP CLI**:
```bash
uv add "mcp[cli]"
```

3. **Setup the server in Claude Desktop**:
```bash
mcp install server.py
```

4. **Run in development mode (for testing)**:
```bash
mcp dev server.py
```

## Usage

### Using the Simulation

1. **Start a New Conversation in Claude Desktop**:
   - Select one of the available prompts:
     - General Conversation
     - Run Simulation
     - Update Simulation Config

2. **Run a Simulation**:
   - Use the "Run Simulation" prompt
   - Specify parameters like population sizes if desired
   - View results in both Claude and the web UI

3. **Modify Simulation Parameters**:
   - Use the specialized prompts to customize behavior
   - Example: "Decrease predator energy cost to 0.5"
   - Changes will be applied to the next simulation run

4. **Analyze Results**:
   - Review the AI-generated analysis
   - Explore population trends in the web UI

### Accessing the Web UI

The web UI is available at:
- http://127.0.0.1:6274 (requires starting a simple HTTP server in the UI directory)

## Available Prompts

1. **General Conversation**: Chat with the assistant
2. **Run Simulation**: Execute the predator-prey simulation
3. **Update Simulation Config**: Add food regeneration to the simulation

## Simulation Parameters

The simulation parameters are defined in `config.yaml`. Key parameters include:

- Initial predator and prey populations
- Number of episodes and steps
- Energy costs for movement
- Nutritional values for feeding and hunting

## Troubleshooting

### Common Issues

1. **AgentTorch Import Errors**:
   - Ensure all dependencies are properly installed
   - Try reinstalling with `pip install git+https://github.com/agenttorch/agenttorch.git`

2. **Claude Desktop Integration Issues**:
   - Verify the MCP server is properly installed: `mcp list`
   - Try reinstalling with `mcp install server.py`

3. **Simulation Runtime Errors**:
   - Check if data files exist in the expected locations
   - Verify your config.yaml is properly formatted

### Useful Commands

```bash
# List installed MCP servers
mcp list

# Reinstall the server
mcp install server.py

# Run with verbose logging
mcp dev server.py --verbose
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.