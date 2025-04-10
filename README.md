# AgentTorch UI Implementation Guide

This guide explains how to set up and run the Grok-inspired UI for the AgentTorch simulation framework.

## Overview

I've created a clean, modern UI inspired by Grok that integrates with your existing AgentTorch MCP server. The interface includes:

1. A chat-like conversation interface
2. Expandable text input for prompts
3. Real-time simulation logs and progress tracking
4. Interactive population dynamics visualization
5. A dark/light theme toggle
6. Tool selection for different types of simulation actions

## Files Structure

- `index.html` - The main HTML file with the UI structure
- `styles.css` - CSS styles for the UI (with dark/light theme support)
- `app.js` - JavaScript code for WebSocket communication and UI interactions
- `server-py-modifications.py` - Additional code to add to your server.py file

## Implementation Steps

### 1. Add the UI Files

Place the `index.html`, `styles.css`, and `app.js` files in a `ui` directory within your project root:

```
project_root/
├── ui/
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── server.py
├── config.yaml
└── ...
```

### 2. Modify server.py

Add the WebSocket server functionality by appending the code from `server-py-modifications.py` to your existing `server.py` file. This adds:

- A WebSocket server that runs alongside your MCP server
- Handlers for simulation, analysis, and conversation requests
- Real-time progress reporting for simulations
- Data conversion for Chart.js visualization

### 3. Update Docker Configuration

Ensure your Docker setup serves the UI files and exposes the WebSocket port:

1. Update your `Dockerfile` to include:
   ```dockerfile
   # Copy UI files
   COPY ui/ /app/ui/
   
   # Expose WebSocket port
   EXPOSE 8765
   ```

2. Update your `docker-compose.yaml` to:
   ```yaml
   ports:
     - "8765:8765"  # WebSocket port
   ```

### 4. Run the Server

Run your server with WebSocket support:

```bash
python server.py websocket
```

Or using Docker:

```bash
docker-compose up
```

## Connecting to Claude Desktop

You can still use your MCP server with Claude Desktop by running:

```bash
mcp install server.py
```

The WebSocket server is specifically for the web UI integration and doesn't affect the MCP protocol compatibility.

## UI Features

### Chat Interface

The UI provides a chat-like interface similar to Grok, with:
- User messages (right-aligned, blue)
- Assistant responses (left-aligned, dark background)
- System messages (center-aligned, gray)

### Tool Selection

Users can select different tools using the toolbar buttons:
- Chat (general conversation)
- Run Simulation (run predator-prey simulation)
- Update Config (modify simulation parameters)
- Analyze Results (analyze simulation outcomes)

### Sample Prompts

The UI includes a sample prompts section with common simulation queries for users to select:
- "What happens to prey population when predators increase?"
- "How does food availability affect the ecosystem balance?"
- "What emergent behaviors appear in predator-prey systems?"
- And more...

### Side Panel

The side panel shows:
- Current simulation status and progress
- Real-time simulation logs
- Population dynamics chart

### Theme Toggle

Users can switch between dark and light themes using the toggle button in the header.

## Customization

### Adding More Tools

To add more tools:

1. Add a new button in the `tools-selector` div in `index.html`
2. Add a handler function in `server-py-modifications.py`
3. Update the tool handling logic in `app.js`

### Changing Chart Appearance

Modify the chart options in the `createPopulationChart` function in `app.js` to customize the appearance of the population dynamics chart.

### Sample Prompts

Update the sample prompts in the `sample-prompts` div in `index.html` to show different example queries.

## Troubleshooting

### WebSocket Connection Issues

If the WebSocket connection fails:
1. Check that port 8765 is exposed and accessible
2. Verify the WebSocket server is running alongside the MCP server
3. Look for error messages in the server logs

### Chart Display Problems

If the chart doesn't display properly:
1. Check the console for JavaScript errors
2. Verify the simulation results contain the expected data format
3. Test with a smaller dataset first

### UI Rendering Issues

If the UI doesn't render correctly:
1. Clear your browser cache
2. Try a different browser
3. Check for CSS conflicts with browser extensions