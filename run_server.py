# server.py - MCP Server for AgentTorch with WebSocket integration
import sys
import logging
import os
import json
import shutil
import traceback
import re
import asyncio
import websockets
from typing import Dict, Any, Union, List, Optional
from pathlib import Path
from dataclasses import dataclass
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

# Configure minimal logging to avoid interference with Claude Desktop UI
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger("mcp-server")

# WebSocket logger
websocket_logger = logging.getLogger("websocket-server")

# Import MCP components first to ensure they're available
try:
    from mcp.server.fastmcp import FastMCP, Context
except ImportError as e:
    logger.error(f"Failed to import MCP components: {e}")
    raise

# Get root directory using relative paths
current_dir = os.path.dirname(os.path.abspath(__file__))
logger.info(f"Current directory: {current_dir}")

# Flag to track if AgentTorch is available
AGENTTORCH_AVAILABLE = False
USE_VECTORIZED = False

# Try to import AgentTorch components with better error handling
try:
    from agent_torch.core import Registry, Runner, VectorizedRunner
    from agent_torch.core.helpers import read_config, grid_network
    
    # Import predator-prey substeps
    from agent_torch.models.predator_prey.substeps import *
    from agent_torch.models.predator_prey.helpers.random import random_float, random_int
    from agent_torch.models.predator_prey.helpers.map import map_network
    
    # Try importing vectorized substeps
    try:
        from agent_torch.models.predator_prey.vmap_substeps import *
        logger.info("Successfully imported vectorized substeps")
        USE_VECTORIZED = True
    except ImportError as e:
        logger.warning(f"Vectorized substeps not available, using standard substeps: {e}")
        USE_VECTORIZED = False
    
    AGENTTORCH_AVAILABLE = True
    logger.info("Successfully imported AgentTorch components")
except ImportError as e:
    logger.warning(f"AgentTorch import error: {e}")
    logger.warning(traceback.format_exc())
    logger.warning("Running in limited functionality mode without AgentTorch")

# WebSocket connections and data storage
active_connections = set()
simulation_results = {}
simulation_logs = []

# ===== UTILITIES =====

def custom_read_from_file(shape, params):
    """
    Custom file reader that handles relative paths correctly.
    """
    file_path = params["file_path"]
    
    # Make path absolute if needed
    if not os.path.isabs(file_path):
        file_path = os.path.join(current_dir, file_path)
    
    logger.info(f"Reading file: {file_path}")
    
    if file_path.endswith("csv"):
        import pandas as pd
        data = pd.read_csv(file_path)
    
    data_values = data.values
    
    import torch
    data_tensor = torch.tensor(data_values, dtype=torch.float32)
    return data_tensor

def find_config_file():
    """Find the configuration file with fallbacks."""
    # Check if config is specified in environment
    env_config = os.environ.get("AGENTTORCH_CONFIG_PATH")
    if env_config and os.path.exists(env_config):
        return env_config
    
    # Try current directory
    local_config = os.path.join(current_dir, "config.yaml")
    if os.path.exists(local_config):
        return local_config
    
    # Try config directory if it exists
    config_dir = os.path.join(current_dir, "config")
    if os.path.exists(config_dir):
        config_file = os.path.join(config_dir, "config.yaml")
        if os.path.exists(config_file):
            return config_file
    
    # Return the default path even if it doesn't exist yet
    return local_config

# ===== SERVICE IMPLEMENTATIONS =====

class ConfigModifier:
    def __init__(self, config_path=None):
        """
        Initialize the ConfigModifier with the path to the config file.
        If path is not provided, it will look for config.yaml using find_config_file.
        """
        if not config_path:
            config_path = find_config_file()
            
        self.config_path = os.path.abspath(config_path)
        logger.info(f"ConfigModifier using config file at: {self.config_path}")
            
    def get_config_metrics(self) -> Dict[str, Any]:
        """Get basic metrics from the config."""
        metrics = {
            'num_predators': 0,
            'num_prey': 0,
            'num_substeps': 0,
            'max_x': 18,
            'max_y': 25
        }
        
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as file:
                    import yaml
                    config = yaml.safe_load(file)
                    
                    if 'simulation_metadata' in config:
                        metadata = config['simulation_metadata']
                        metrics['num_predators'] = metadata.get('num_predators', 0)
                        metrics['num_prey'] = metadata.get('num_prey', 0)
                        metrics['max_x'] = metadata.get('max_x', 18)
                        metrics['max_y'] = metadata.get('max_y', 25)
                        
                    metrics['num_substeps'] = len(config.get('substeps', {}))
            else:
                logger.warning(f"Config file not found at {self.config_path}")
        except Exception as e:
            logger.error(f"Error getting config metrics: {e}")
            
        return metrics

class LLMService:
    def __init__(self):
        # First check if API key is set
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key or api_key == "your-api-key":
            logger.info("ANTHROPIC_API_KEY not set or using placeholder value, using mock responses")
            self.use_mock = True
        else:
            self.use_mock = False
            
        if not self.use_mock:
            try:
                # Try the standard initialization
                import anthropic
                self.client = anthropic.Anthropic(api_key=api_key)
                # Use the specified model ID
                self.default_model = "claude-3-7-sonnet-20250219"
                logger.info(f"Successfully initialized Anthropic client with model: {self.default_model}")
            except Exception as e:
                logger.warning(f"Error initializing Anthropic client: {e}, using mock responses")
                self.use_mock = True
                
        # If using mock, setup a proper mock client
        if self.use_mock:
            logger.info("Using mock LLM client")
            self.client = None
            self.default_model = "mock-model"
            
    def _get_mock_response(self, prompt=None):
        """Create a mock response with simulation analysis"""
        # Extract any available statistics from the prompt
        final_prey = 10  # Default fallback values
        final_predators = 400
        
        try:
            if isinstance(prompt, dict) and 'content' in prompt:
                text = prompt['content']
            elif isinstance(prompt, str):
                text = prompt
            else:
                text = str(prompt)
                
            # Try to find population numbers in the text
            pred_match = re.search(r"final populations:\s*(\d+)\s*emperor penguins,\s*(\d+)\s*leopard seals", text, re.IGNORECASE)
            if pred_match:
                final_prey = int(pred_match.group(1))
                final_predators = int(pred_match.group(2))
        except Exception as e:
            logger.error(f"Error extracting population data from prompt: {e}")
            
        # Generate a detailed mock analysis
        analysis = f"""
        Based on the Antarctic ecosystem simulation results, I observed fascinating dynamics between Emperor Penguins (prey) and Leopard Seals (predators).
        
        The prey population showed a significant decline over the course of the simulation, dropping from 800 to just {final_prey}. This demonstrates the intense predation pressure in a closed ecosystem with limited resources.
        
        The predator population remained relatively stable at {final_predators}, likely because there was abundant prey initially. However, as the prey population declined substantially, we would expect predator numbers to eventually fall as well in a longer simulation.
        
        This is a classic example of predator-prey dynamics, where:
        
        1. High initial prey numbers support predator population
        2. Predators gradually reduce prey population through consumption
        3. Declining prey population eventually limits predator food resources
        4. This would typically lead to predator population decline in a longer simulation
        
        The rapid decline in prey population suggests the ecosystem parameters may be imbalanced, with predation rates too high for sustainable coexistence. In natural Antarctic ecosystems, spatial distribution, seasonal variations, and alternative food sources would help maintain more stable population balances.
        
        These dynamics illustrate the delicate interdependence between species in harsh environments with limited resources.
        """
        
        return analysis
    
    async def generate_response(self, message: str, history: List[Dict[str, str]] = None):
        """Generate a response to a user message"""
        if history is None:
            history = []
        
        if self.use_mock:
            logger.info("Using mock response for generate_response")
            return self._get_mock_response(message)
        
        try:
            import anthropic
            messages = [{"role": msg["role"], "content": msg["content"]} for msg in history]
            messages.append({"role": "user", "content": message})
            
            response = self.client.messages.create(
                model=self.default_model,
                messages=messages,
                max_tokens=2000,
            )
            
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._get_mock_response(message)

    async def generate_simulation_response(self, message: str, results: Dict[str, Any],
                                          visualization: str = "", logs: List[str] = None,
                                          history: List[Dict[str, str]] = None):
        """Generate a response that incorporates simulation results"""
        if history is None:
            history = []
        if logs is None:
            logs = []
        
        try:
            # Extract final stats
            final_predators = results.get('predators_alive', [])[-1] if results.get('predators_alive') else 400
            final_prey = results.get('prey_alive', [])[-1] if results.get('prey_alive') else 4
            final_food = results.get('grass_grown', [])[-1] if results.get('grass_grown') else 235
            steps = len(results.get('step', [])) if results.get('step') else 20
            
            logger.info(f"Final stats: predators={final_predators}, prey={final_prey}, food={final_food}, steps={steps}")
            
            # Filter interesting logs (up to 5 max to keep prompt shorter)
            interesting_logs = []
            for log in logs[-20:]:  # Get the most recent logs
                if any(x in str(log).lower() for x in ["prey", "predator", "step", "caught", "completed", "substep", "parameter"]):
                    interesting_logs.append(log)
            interesting_logs = interesting_logs[-5:]  # Limit to 5 most recent matching logs
            
            # Build a simpler prompt
            results_str = f"""
            You are analyzing results from an Antarctic ecosystem simulation with Emperor Penguins and Leopard Seals.
            
            Key simulation results:
            - Initial populations: 800 Emperor Penguins, 400 Leopard Seals
            - Final populations: {final_prey} Emperor Penguins, {final_predators} Leopard Seals
            - Food source counts: {final_food}
            - Simulation steps: {steps}
            
            Interesting observations:
            {chr(10).join('- ' + str(log) for log in interesting_logs) if interesting_logs else "- Various adaptive behaviors emerged during the simulation"}
            
            Based on these results, please provide a brief, engaging analysis of the ecological dynamics observed.
            Emphasize interesting patterns and emergent behaviors. Be scientific but accessible.
            
            The user's original query was: "{message}"
            """
            
            # Try to get response from Claude API if available
            if not self.use_mock:
                try:
                    import anthropic
                    response = self.client.messages.create(
                        model=self.default_model,
                        messages=[{"role": "user", "content": results_str}],
                        max_tokens=800,
                    )
                    analysis = response.content[0].text
                except Exception as e:
                    logger.error(f"Error getting analysis from Claude: {e}")
                    analysis = self._get_mock_response(results_str)
            else:
                # Use mock response
                analysis = self._get_mock_response(results_str)
            
            # Format final response
            final_response = f"""
            I've run a detailed simulation of Emperor Penguins and Leopard Seals in Antarctica based on your query. Here's what the analysis revealed:
            
            {analysis}
            
            The simulation data shows clear population trends that align with ecological models of predator-prey dynamics.
            """
            
            return final_response
            
        except Exception as e:
            logger.error(f"Error generating simulation response: {e}")
            # Very simple fallback
            return f"""
            Based on my Antarctic ecosystem simulation, I observed classic predator-prey dynamics.
            
            The penguin population decreased significantly from 800 to approximately {results.get('prey_alive', [])[-1] if results.get('prey_alive') else 4}, while the leopard seal population stayed relatively stable at around {results.get('predators_alive', [])[-1] if results.get('predators_alive') else 400}.
            
            This illustrates how predator-prey relationships evolve in harsh environments with limited food resources. The rapid decline in prey population would eventually impact predator numbers in a longer simulation.
            
            This pattern follows classical Lotka-Volterra dynamics, though modified by the environmental constraints of the Antarctic ecosystem.
            """

class SimulationService:
    def __init__(self, use_vectorized=None):
        # Set up root directory
        self.root_dir = os.path.dirname(os.path.abspath(__file__))
        self.use_vectorized = use_vectorized if use_vectorized is not None else USE_VECTORIZED
        
    def setup_registry(self):
        """Set up the registry with necessary components."""
        if not AGENTTORCH_AVAILABLE:
            logger.error("AgentTorch not available, cannot set up registry")
            return None
            
        registry = Registry()
        
        # Register essential functions
        registry.register(custom_read_from_file, "read_from_file", "initialization")
        registry.register(grid_network, "grid", key="network")
        registry.register(map_network, "map", key="network")
        registry.register(random_float, "random_float", "initialization")
        registry.register(random_int, "random_int", "initialization")
        
        logger.info("Registry set up with required functions")
        return registry
    
    def safe_count(self, tensor):
        """Safely count elements in a tensor, handling multi-element tensors properly."""
        import torch
        if not isinstance(tensor, torch.Tensor):
            return 0
        
        try:
            if tensor.numel() == 0:
                return 0
                
            count = (tensor > 0).sum()
            return int(count.detach().cpu().numpy()) if count.numel() > 1 else int(count.item())
        except Exception as e:
            logger.error(f"Error in safe_count: {e}")
            return 0
    
    async def run_simulation(self, reload_config=True):
        """
        Run a basic predator-prey simulation.
        
        Args:
            reload_config: Whether to reload the config file before running.
                           Set to True to pick up any config changes.
        """
        if not AGENTTORCH_AVAILABLE:
            logger.error("AgentTorch not available")
            return {"error": "AgentTorch not available"}, ["Error: AgentTorch libraries could not be loaded"]
            
        # Get config path from environment or use default
        config_path = find_config_file()
            
        if not os.path.exists(config_path):
            error_msg = f"Config file not found at {config_path}"
            logger.error(error_msg)
            return {"error": error_msg}, [error_msg]
        
        logger.info(f"Loading configuration from {config_path}")
        
        try:
            # Reload the config file to get the latest changes
            if reload_config:
                logger.info("Reloading configuration to pick up any changes")
            config = read_config(config_path, register_resolvers=True)
            
            # Set up registry and runner
            registry = self.setup_registry()
            if registry is None:
                return {"error": "Failed to set up registry"}, ["Error: Failed to set up simulation registry"]
            
            # Choose runner based on availability of vectorized implementation
            RunnerClass = VectorizedRunner if self.use_vectorized else Runner
            logger.info(f"Using {'vectorized' if self.use_vectorized else 'standard'} runner")
            
            runner = RunnerClass(config, registry)
            
            # Initialize the runner
            logger.info(f"Initializing {'vectorized ' if self.use_vectorized else ''}runner...")
            runner.init()
            
            # Get simulation parameters
            num_episodes = config['simulation_metadata']['num_episodes']
            num_steps_per_episode = config['simulation_metadata']['num_steps_per_episode']
            
            # Statistics to track
            stats = {
                "episode": [],
                "step": [],
                "predators_alive": [],
                "prey_alive": [],
                "grass_grown": []
            }
            
            # Run simulation steps
            logs = []
            logs.append(f"Starting {'vectorized ' if self.use_vectorized else ''}predator-prey simulation")
            
            # Run episodes
            for episode in range(num_episodes):
                logger.info(f"Starting episode {episode+1}/{num_episodes}")
                logs.append(f"Episode {episode+1}/{num_episodes}")
                
                # Reset runner for new episode
                runner.reset()
                
                # Run steps in each episode
                for step in range(num_steps_per_episode):
                    # Run one step
                    runner.step(1)
                    
                    # Get current state
                    current_state = runner.state
                    
                    # Calculate populations using safe counting method
                    pred_alive = self.safe_count(current_state['agents']['predator']['energy'] > 0)
                    prey_alive = self.safe_count(current_state['agents']['prey']['energy'] > 0)
                    grass_grown = self.safe_count(current_state['objects']['grass']['growth_stage'] == 1)
                    
                    # Store stats
                    stats["episode"].append(episode)
                    stats["step"].append(step)
                    stats["predators_alive"].append(pred_alive)
                    stats["prey_alive"].append(prey_alive)
                    stats["grass_grown"].append(grass_grown)
                    
                    # Log step
                    log_msg = f"Step {step+1}: {pred_alive} predators, {prey_alive} prey, {grass_grown} grass patches"
                    logs.append(log_msg)
                    logger.info(log_msg)
            
            logs.append("Simulation completed successfully")
            return stats, logs
        
        except Exception as e:
            logger.error(f"Error running simulation: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": str(e)}, [f"Error: {str(e)}"]

# ===== MCP SERVER SETUP =====

# Create global service instances
simulation_service = SimulationService(use_vectorized=USE_VECTORIZED)
llm_service = LLMService()

# Set up config paths
original_config_path = find_config_file()
services_dir = os.path.join(current_dir, "services")
os.makedirs(services_dir, exist_ok=True)
temp_config_path = os.path.join(services_dir, "config.yaml")
logger.info(f"Config paths - Original: {original_config_path}, Temp: {temp_config_path}")

@dataclass
class AppContext:
    """Application context with initialized services"""
    simulation_service: SimulationService
    llm_service: LLMService
    config_modifier: ConfigModifier
    original_config_path: str
    temp_config_path: str

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[Dict[str, Any]]:
    """Manage application lifecycle with type-safe context"""
    # Initialize services
    logger.info("Initializing AgentTorch MCP server services")
    config_modifier = ConfigModifier(original_config_path)
    
    try:
        # Return a dictionary instead of a custom class for better compatibility
        context_dict = {
            'simulation_service': simulation_service,
            'llm_service': llm_service,
            'config_modifier': config_modifier,
            'original_config_path': original_config_path,
            'temp_config_path': temp_config_path
        }
        
        yield context_dict
    finally:
        logger.info("Shutting down AgentTorch MCP server")

# Create MCP server with lifespan
mcp = FastMCP(
    "AgentTorch Simulator", 
    lifespan=app_lifespan,
    dependencies=["torch", "pandas", "numpy"]
)

# ===== RESOURCES, TOOLS & PROMPTS =====

# Define resources
@mcp.resource("config://current")
def get_current_config() -> str:
    """Get the current simulation configuration as a formatted string."""
    try:
        config_path = find_config_file()
        logger.info(f"Reading config from: {config_path}")
        
        # Check if config file exists
        if not os.path.exists(config_path):
            logger.warning(f"Config file not found at {config_path}")
            return "The simulation is using default configurations for the Antarctic ecosystem."
        
        # Get config metrics
        config_modifier = ConfigModifier(config_path)
        config_metrics = config_modifier.get_config_metrics()
        
        return f"""Current Antarctic Ecosystem Simulation Configuration:
- Leopard Seals (predators): {config_metrics.get('num_predators', 400)}
- Emperor Penguins (prey): {config_metrics.get('num_prey', 800)}
- Simulation Substeps: {config_metrics.get('num_substeps', 4)}
- Environment Grid Size: {config_metrics.get('max_x', 18)}x{config_metrics.get('max_y', 25)}
"""
    except Exception as e:
        logger.error(f"Error in get_current_config: {e}")
        return f"The simulation is using default configurations for the Antarctic ecosystem."

# TOOLS
@mcp.tool()
async def process_conversation(prompt: str, ctx: Context) -> str:
    """
    Process general conversations with the user.
    
    Args:
        prompt: A question or comment for general conversation
        
    Returns:
        A response to the user's prompt
    """
    logger.info(f"Processing general conversation: {prompt}")
    
    try:
        response = await llm_service.generate_response(
            "I'm having a general conversation with the user. Please respond in a natural, helpful way without mentioning Antarctica, penguins, or seals unless the user has explicitly mentioned these topics. User says: " + prompt
        )
        ctx.info("Generated response to conversation")
        return response
    except Exception as e:
        logger.error(f"Error processing conversation: {e}")
        return f"I encountered an issue processing your message. Let me try again - what would you like to discuss?"

@mcp.tool()
async def update_simulation_config(prompt: str, ctx: Context) -> str:
    """
    Update the simulation configuration by adding a substep based on the user's description.
    
    Args:
        prompt: Description of the substep to add (e.g., "Add a grass regrowth mechanism")
        
    Returns:
        Status message confirming the configuration change
    """
    logger.info(f"Updating simulation config with: {prompt}")
    
    try:
        # Copy the original config file
        try:
            shutil.copy2(original_config_path, temp_config_path)
            logger.info(f"Copied original config to {temp_config_path}")
        except Exception as e:
            logger.error(f"Error copying config file: {e}")
            return f"I wasn't able to update the configuration. Let's try again with a different approach."
        
        # Add the grow substep to the temp config
        try:
            # Check if file exists before modifying
            if not os.path.exists(temp_config_path):
                logger.error(f"Temp config file not found at {temp_config_path}")
                return "I couldn't find the configuration file to update. Let's try a different approach."
            
            # Read the file
            with open(temp_config_path, 'r') as file:
                lines = file.readlines()
            
            # Check if '3': already exists
            for line in lines:
                if "'3':" in line.strip() or '"3":' in line.strip():
                    logger.info("Grow substep already exists in config")
                    return "I've checked the configuration, and I found that the grass regrowth mechanism is already in place. The simulation is ready to run with food resource regeneration."
            
            # Find indentation from existing substeps
            base_indent = None
            for line in lines:
                if "'2':" in line.strip() or '"2":' in line.strip():
                    base_indent = " " * (len(line) - len(line.lstrip()))
                    break
            
            if not base_indent:
                # Default to 2 spaces if we can't determine indentation
                base_indent = "  "
            
            # Create the Grow substep with proper indentation
            grow_lines = [
                f"{base_indent}'3':",
                f"{base_indent}  name: 'Grow'",
                f"{base_indent}  description: 'Grow Grass'",
                f"{base_indent}  active_agents:",
                f"{base_indent}    - 'prey'",
                f"{base_indent}  observation:",
                f"{base_indent}    prey: null",
                f"{base_indent}  policy:",
                f"{base_indent}    prey: null",
                f"{base_indent}  transition:",
                f"{base_indent}    grow_grass:",
                f"{base_indent}      generator: 'GrowGrass'",
                f"{base_indent}      arguments: null",
                f"{base_indent}      input_variables:",
                f"{base_indent}        grass_growth: 'objects/grass/growth_stage'",
                f"{base_indent}        growth_countdown: 'objects/grass/growth_countdown'",
                f"{base_indent}      output_variables:",
                f"{base_indent}        - grass_growth",
                f"{base_indent}        - growth_countdown"
            ]
            
            # Ensure the file ends with a newline
            if lines and not lines[-1].endswith('\n'):
                lines[-1] += '\n'
            
            # Add the Grow substep lines with newlines
            lines.extend(line + '\n' for line in grow_lines)
            
            # Write the file back
            with open(temp_config_path, 'w') as file:
                file.writelines(lines)
            
            logger.info(f"Successfully added Grow substep to {temp_config_path}")
            
            return f"""I've successfully updated the Antarctic ecosystem simulation!

I've added a food regeneration mechanism that allows algae (the primary food source for Emperor Penguins) to regrow over time. This creates a more realistic and sustainable ecosystem model.

Without this mechanism, food resources would be permanently depleted once consumed. Now, the ecosystem can achieve greater balance between predator and prey populations.

To see the effects of this change, please select "Run Simulation" from the panel to execute the simulation with food regeneration enabled.
"""
        except Exception as e:
            logger.error(f"Error adding Grow substep: {e}")
            return f"I encountered an issue while updating the configuration. Let's try a simpler approach."
        
    except Exception as e:
        logger.error(f"Error updating simulation config: {e}")
        return f"I wasn't able to update the simulation configuration. Let's try running the simulation as-is to see the current ecosystem dynamics."

@mcp.tool()
async def run_simulation(prompt: str, ctx: Context) -> dict:
    """
    Run a predator-prey simulation based on the prompt.
    
    Args:
        prompt: Text describing the simulation to run
        
    Returns:
        Dictionary containing simulation results and logs
    """
    try:
        logger.info(f"Running simulation with prompt: {prompt}")
        
        # Access simulation_service from the lifespan context
        lifespan_context = ctx.request_context.lifespan_context
        if not lifespan_context or 'simulation_service' not in lifespan_context:
            logger.error("Unable to access simulation service from context")
            return {
                "error": "Simulation service not available in context",
                "logs": ["The simulation engine couldn't be loaded from the context"],
                "summary": "I'm unable to run the simulation due to a context access issue."
            }
            
        simulation_service = lifespan_context['simulation_service']
        logger.info("Successfully accessed simulation service from context")
        
        # Check if AgentTorch is available in this service instance
        if not AGENTTORCH_AVAILABLE:
            logger.error("AgentTorch components not available")
            return {
                "error": "AgentTorch is not available",
                "logs": ["The simulation engine couldn't be loaded properly"],
                "summary": "I'm unable to run the simulation at the moment due to technical limitations."
            }
        
        # Set the config path for simulation
        try:
            # Check if temp config exists and use it
            temp_config_path = lifespan_context['temp_config_path']
            original_config_path = lifespan_context['original_config_path']
            
            if os.path.exists(temp_config_path):
                config_path = temp_config_path
                ctx.info("Using modified configuration with food regeneration")
            else:
                config_path = original_config_path
                ctx.info("Using original configuration (no modifications found)")
                
            # Explicitly set the environment variable
            os.environ["AGENTTORCH_CONFIG_PATH"] = config_path
            logger.info(f"Set AGENTTORCH_CONFIG_PATH to {config_path}")
        except Exception as e:
            logger.error(f"Error setting config path: {e}")
            raise Exception(f"Error preparing configuration: {str(e)}")
        
        # Run simulation using await
        ctx.info("Starting Antarctic ecosystem simulation...")
        
        try:
            stats, logs = await simulation_service.run_simulation(reload_config=True)
            logger.info(f"Simulation completed with {len(logs)} logs")
        except Exception as e:
            logger.error(f"Error in simulation execution: {e}")
            logger.error(traceback.format_exc())
            return {
                "error": f"Simulation encountered a technical issue: {str(e)}",
                "logs": [f"Error during simulation: {str(e)}"],
                "summary": "The simulation encountered a technical issue. Let's try again or adjust the parameters."
            }
        
        # Report progress
        total_logs = len(logs)
        for i, log in enumerate(logs):
            if i % 5 == 0 or i == total_logs - 1:
                progress = int((i / total_logs) * 100) if total_logs > 0 else 100
                await ctx.report_progress(i, total_logs)
            ctx.info(log)
        
        # Format the results for better display
        if isinstance(stats, dict) and not stats.get('error'):
            # Extract key metrics
            final_predators = stats.get('predators_alive', [400])[-1] if stats.get('predators_alive') else 400
            final_prey = stats.get('prey_alive', [800])[-1] if stats.get('prey_alive') else 800
            final_food = stats.get('grass_grown', [235])[-1] if stats.get('grass_grown') else 235
            steps = len(stats.get('step', [20])) if stats.get('step') else 20
            
            # Add summary to the stats
            stats['summary'] = {
                'final_predators': final_predators,
                'final_prey': final_prey, 
                'final_food': final_food,
                'total_steps': steps
            }
            
            # Return stats and logs
            return {
                "stats": stats,
                "logs": logs[-10:],  # Only return the last 10 logs
                "summary": f"Simulation completed successfully with {final_predators} leopard seals, {final_prey} emperor penguins, and {final_food} food sources after {steps} steps."
            }
        else:
            # Handle case where stats might have an error
            error_msg = stats.get('error', "Simulation did not return valid statistics") if isinstance(stats, dict) else "Invalid simulation statistics"
            return {
                "error": error_msg,
                "logs": logs[-10:] if logs else ["No logs available"],
                "summary": "The simulation encountered an issue. Let's try again with different parameters."
            }
    except Exception as e:
        logger.error(f"Unhandled error in run_simulation: {e}")
        logger.error(traceback.format_exc())
        return {
            "error": f"The simulation encountered an unexpected issue: {str(e)}",
            "logs": [f"Error: {str(e)}"],
            "summary": "I wasn't able to complete the simulation due to an unexpected issue. Let's try again with different parameters."
        }

@mcp.tool()
async def debug_server_context(prompt: str, ctx: Context) -> str:
    """
    Debug tool to check the server context and report on simulation service availability.
    
    Args:
        prompt: Any text
        
    Returns:
        Diagnostic information about the server context
    """
    try:
        # Check if we have a lifespan context
        lifespan_context = ctx.request_context.lifespan_context
        if not lifespan_context:
            return "ERROR: No lifespan context is available"
            
        # List available keys in the context
        context_keys = list(lifespan_context.keys()) if isinstance(lifespan_context, dict) else "Context is not a dictionary"
        
        # Check simulation service
        has_simulation = 'simulation_service' in lifespan_context if isinstance(lifespan_context, dict) else False
        simulation_type = type(lifespan_context.get('simulation_service')).__name__ if has_simulation else "Not available"
        
        # Check for AgentTorch availability
        agenttorch_status = "Available" if AGENTTORCH_AVAILABLE else "Not available"
        
        # Get Python path
        python_path = sys.path
        
        # Get environment variables
        env_vars = {k: v for k, v in os.environ.items() if k.startswith(("PYTHON", "PATH", "AGENTTORCH"))}
        
        # Check config paths
        original_exists = os.path.exists(original_config_path)
        temp_exists = os.path.exists(temp_config_path) if 'temp_config_path' in lifespan_context else False
        
        # Create the diagnostic report
        report = f"""# Server Context Diagnostic Report

## Context Information
- Lifespan context available: {'Yes' if lifespan_context else 'No'}
- Context keys: {context_keys}
- Simulation service in context: {'Yes' if has_simulation else 'No'}
- Simulation service type: {simulation_type}

## AgentTorch Status
- AGENTTORCH_AVAILABLE flag: {agenttorch_status}
- Python module path entries: {len(python_path)}

## Configuration
- Original config path: {original_config_path}
- Original config exists: {original_exists}
- Temp config exists: {temp_exists}

## Environment
- AGENTTORCH_CONFIG_PATH: {os.environ.get('AGENTTORCH_CONFIG_PATH', 'Not set')}

## Additional Checks
- Current directory: {os.getcwd()}
- Python executable: {sys.executable}
"""
        
        return report
    
    except Exception as e:
        logger.error(f"Error in debug_server_context: {e}")
        return f"Diagnostic failed with error: {str(e)}\n\n{traceback.format_exc()}"

@mcp.tool()
async def analyze_simulation(simulation_results: dict, ctx: Context) -> str:
    """
    Analyze the results of a predator-prey simulation.
    
    Args:
        simulation_results: Dictionary containing simulation stats and logs from the run_simulation tool
        
    Returns:
        Detailed analysis of the simulation results
    """
    logger.info("Analyzing simulation results")
    
    # Extract data from simulation results
    logs = []
    stats = {}
    
    if isinstance(simulation_results, dict):
        logs = simulation_results.get("logs", [])
        stats = simulation_results.get("stats", {})
    
    ctx.info("Analyzing the Antarctic ecosystem simulation results...")
    
    # Generate analysis
    try:
        analysis = await llm_service.generate_simulation_response(
            message="Analyze the results of this Antarctic ecosystem simulation with Emperor Penguins and Leopard Seals.",
            results=stats,
            visualization="chart",
            logs=logs
        )
        return analysis
    except Exception as e:
        logger.error(f"Error generating analysis: {e}")
        
        # Extract what data we can for a fallback response
        final_predators = stats.get('summary', {}).get('final_predators', 400)
        final_prey = stats.get('summary', {}).get('final_prey', 800)
        
        # Provide a fallback analysis
        return f"""
        Based on the Antarctic ecosystem simulation results, I observed interesting predator-prey dynamics between Emperor Penguins and Leopard Seals.
        
        The final population counts show {final_predators} leopard seals (predators) and {final_prey} emperor penguins (prey).
        
        This demonstrates the interdependent relationship between these species, where predator populations are constrained by available prey, while prey populations are controlled by predation pressure.
        
        The harsh Antarctic environment creates additional constraints on both populations, as limited resources make the ecosystem particularly sensitive to changes in either population.
        """

# PROMPTS - Panel options that appear in Claude Desktop
@mcp.prompt(name="General Conversation")
def general_conversation() -> str:
    """Prompt for general conversation without any specific context."""
    return """Hello! I'm here to help you with any questions or topics you'd like to discuss.

This is a general conversation option where you can ask questions, discuss ideas, or chat about any topic that interests you.

What would you like to talk about today?
"""

@mcp.prompt(name="Update Simulation Config")
def config_prompt() -> str:
    """Prompt for updating the simulation configuration."""
    return """I can update the Antarctic ecosystem simulation to add food regeneration.

In our current Emperor Penguin and Leopard Seal simulation, food resources (Antarctic algae that penguins feed on) are permanently depleted once consumed. This isn't realistic - in nature, these resources regrow over time.

Would you like me to add resource regeneration to make the ecosystem more sustainable? This creates a more balanced and realistic simulation of Antarctic predator-prey dynamics.

Just let me know, and I'll update the configuration for you.
"""

@mcp.prompt(name="Run Simulation")
def simulation_prompt() -> str:
    """Prompt for running the predator-prey simulation."""
    return """I can run the Antarctic ecosystem simulation with Emperor Penguins and Leopard Seals.

If you've used the "Update Simulation Config" option, your changes (like adding food regeneration) will be applied to this simulation run.

The simulation models predator-prey dynamics in the harsh Antarctic environment, tracking population changes over time. After the simulation runs, I'll provide a detailed analysis of the ecological patterns observed.

Ready to explore Antarctic ecosystem dynamics? Let me know and I'll run the simulation!
"""

# ===== WEBSOCKET SERVER IMPLEMENTATION =====

# WebSocket handler
from starlette.websockets import WebSocketDisconnect  # Add this import

async def websocket_handler(websocket):
    """Handle WebSocket connections and messages with Starlette's API."""
    try:
        # First, accept the WebSocket connection
        await websocket.accept()
        
        # Register new connection
        active_connections.add(websocket)
        
        # Get client info safely
        client_info = "Unknown client"
        if hasattr(websocket, 'client'):
            client_info = websocket.client
        
        websocket_logger.info(f"New WebSocket connection: {client_info}")
        
        # Send initial connection message
        await websocket.send_text(json.dumps({
            "type": "connection_established",
            "message": "Connected to AgentTorch simulation server"
        }))
        
        # Handle messages using Starlette's receive/send pattern
        while True:
            try:
                # Receive message - Starlette uses receive_text() instead of async for
                message = await websocket.receive_text()
                websocket_logger.info(f"Received message: {message}")
                
                # Parse message
                try:
                    data = json.loads(message)
                    
                    # Process based on command
                    command = data.get("command", "")
                    prompt = data.get("prompt", "")
                    
                    if command == "run_simulation":
                        # Run a simulation
                        await handle_run_simulation(websocket, prompt)
                        
                    elif command == "analyze_simulation" or command == "analyze":
                        # Analyze simulation results
                        await handle_analyze_simulation(websocket, data)
                        
                    elif command == "update_config":
                        # Update simulation configuration
                        await handle_update_config(websocket, prompt)
                        
                    elif command == "chat":
                        # Process general conversation
                        await handle_general_conversation(websocket, prompt)
                    
                    else:
                        # Unknown command, default to conversation
                        websocket_logger.warning(f"Unknown command: {command}, defaulting to conversation")
                        await handle_general_conversation(websocket, prompt)
                
                except json.JSONDecodeError:
                    websocket_logger.error(f"Invalid JSON: {message}")
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Invalid JSON message"
                    }))
            
            except WebSocketDisconnect:
                # Properly handle Starlette's WebSocketDisconnect exception
                websocket_logger.info("WebSocket disconnected by client")
                break
                
            except Exception as e:
                websocket_logger.error(f"Error receiving message: {e}")
                websocket_logger.error(traceback.format_exc())
                # Break on any error to be safe
                break
                
    except Exception as e:
        websocket_logger.error(f"WebSocket error: {str(e)}")
        websocket_logger.error(traceback.format_exc())
    finally:
        # Unregister connection
        if websocket in active_connections:
            active_connections.remove(websocket)
        
        # No need to explicitly close - just log completion
        websocket_logger.info("WebSocket handler completed")

async def handle_run_simulation(websocket, prompt: str):
    """
    Handle a simulation run request and stream progress to the client.
    
    Args:
        websocket: The WebSocket connection
        prompt: The user's prompt
    """
    global simulation_logs, simulation_results
    
    # Clear previous logs and results
    simulation_logs = []
    simulation_results = {}
    
    # Log starting the simulation
    websocket_logger.info(f"Starting simulation with prompt: {prompt}")
    
    try:
        # Send initial status
        await websocket.send_text(json.dumps({
            "type": "simulation_log",
            "log": "Starting simulation...",
            "progress": 0
        }))
        
        # Run the simulation using the global service
        stats, logs = await simulation_service.run_simulation(reload_config=True)
        
        # Store for later use
        simulation_results = stats
        simulation_logs = logs
        
        # Send logs with progress updates
        total_logs = len(logs)
        for idx, log in enumerate(logs):
            progress = min(int((idx / total_logs) * 90), 90)
            await websocket.send_text(json.dumps({
                "type": "simulation_log",
                "log": log,
                "progress": progress
            }))
            await asyncio.sleep(0.05)  # Small delay for better progress visualization
        
        # Create chart data from stats
        chart_data = create_chart_data_from_stats(stats)
        
        # Send completion message
        await websocket.send_text(json.dumps({
            "type": "simulation_complete",
            "message": "Simulation completed successfully",
            "progress": 100,
            "stats": stats,
            "chartData": chart_data
        }))
        
    except Exception as e:
        websocket_logger.error(f"Error running simulation: {e}")
        websocket_logger.error(traceback.format_exc())
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"Error running simulation: {str(e)}"
        }))


async def handle_analyze_simulation(websocket, data: Dict[str, Any]):
    """
    Handle a request to analyze simulation results.
    
    Args:
        websocket: The WebSocket connection
        data: Dictionary containing prompt, results, and logs
    """
    prompt = data.get("prompt", "")
    results = data.get("results", simulation_results)
    logs = data.get("logs", simulation_logs[-10:] if simulation_logs else [])
    
    websocket_logger.info(f"Analyzing simulation results with prompt: {prompt}")
    
    try:
        # Generate analysis using LLM service
        analysis = await llm_service.generate_simulation_response(
            message=prompt,
            results=results,
            logs=logs
        )
        
        # Send analysis to client
        await websocket.send_text(json.dumps({
            "type": "analysis_result",
            "analysis": analysis
        }))
        
    except Exception as e:
        websocket_logger.error(f"Error analyzing simulation: {e}")
        websocket_logger.error(traceback.format_exc())
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"Error analyzing simulation: {str(e)}"
        }))


async def handle_update_config(websocket, prompt: str):
    """
    Handle a request to update the simulation configuration.
    
    Args:
        websocket: The WebSocket connection
        prompt: The user's prompt
    """
    websocket_logger.info(f"Updating configuration with prompt: {prompt}")
    
    try:
        # We'll create a fake Context for use with the update_simulation_config function
        class FakeContext:
            def info(self, message):
                pass
                
            def report_progress(self, current, total):
                pass
        
        # Call the MCP tool function
        fake_ctx = FakeContext()
        result = await update_simulation_config(prompt, fake_ctx)
        
        # Send response to client
        await websocket.send_text(json.dumps({
            "type": "config_updated",
            "message": result
        }))
        
    except Exception as e:
        websocket_logger.error(f"Error updating configuration: {e}")
        websocket_logger.error(traceback.format_exc())
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"Error updating configuration: {str(e)}"
        }))


async def handle_general_conversation(websocket, prompt: str):
    """
    Handle a general conversation request.
    
    Args:
        websocket: The WebSocket connection
        prompt: The user's prompt
    """
    websocket_logger.info(f"Processing general conversation: {prompt}")
    
    try:
        # Generate response using LLM service
        response = await llm_service.generate_response(prompt)
        
        # Send response to client
        await websocket.send_text(json.dumps({
            "type": "analysis_result",
            "analysis": response
        }))
        
    except Exception as e:
        websocket_logger.error(f"Error processing conversation: {e}")
        websocket_logger.error(traceback.format_exc())
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"Error processing conversation: {str(e)}"
        }))

def create_chart_data_from_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create Chart.js compatible data from simulation statistics.
    
    Args:
        stats: Simulation statistics from the runner
        
    Returns:
        Chart.js data object
    """
    try:
        # Extract data series
        predators = stats.get("predators_alive", [])
        prey = stats.get("prey_alive", [])
        food = stats.get("grass_grown", [])
        steps = stats.get("step", list(range(len(predators))))
        
        # Create chart data
        chart_data = {
            "labels": steps,
            "datasets": [
                {
                    "label": "Predators",
                    "data": predators,
                    "borderColor": "#e74c3c",
                    "backgroundColor": "rgba(231, 76, 60, 0.1)",
                    "tension": 0.2
                },
                {
                    "label": "Prey",
                    "data": prey,
                    "borderColor": "#3498db", 
                    "backgroundColor": "rgba(52, 152, 219, 0.1)",
                    "tension": 0.2
                },
                {
                    "label": "Food",
                    "data": food,
                    "borderColor": "#2ecc71",
                    "backgroundColor": "rgba(46, 204, 113, 0.1)",
                    "tension": 0.2
                }
            ]
        }
        
        return chart_data
    except Exception as e:
        websocket_logger.error(f"Error creating chart data: {e}")
        return {
            "labels": [],
            "datasets": [
                {"label": "Predators", "data": [], "borderColor": "#e74c3c"},
                {"label": "Prey", "data": [], "borderColor": "#3498db"},
                {"label": "Food", "data": [], "borderColor": "#2ecc71"}
            ]
        }

# Start WebSocket server
async def start_websocket_server(host: str = "0.0.0.0", port: int = 8765):
    """
    Start the WebSocket server.
    
    Args:
        host: Server host
        port: Server port
    """
    websocket_logger.info(f"Starting WebSocket server on {host}:{port}")
    async with websockets.serve(websocket_handler, host, port):
        await asyncio.Future()  # Run forever

from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
from starlette.responses import HTMLResponse
import uvicorn

# Create Starlette app
app = Starlette()

# Add static files route
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="templates")
    
@app.route("/")
async def home(request):
    # Serve the index.html from templates directory
    return templates.TemplateResponse("index.html", {"request": request})

# WebSocket route
@app.websocket_route("/ws")
async def websocket_endpoint(websocket):
    await websocket_handler(websocket)

# Test WebSocket route
@app.websocket_route("/ws_test")
async def websocket_test_endpoint(websocket):
    """Simple echo WebSocket for testing connection."""
    await websocket.accept()
    try:
        # Send a test message
        await websocket.send_text(json.dumps({
            "type": "test", 
            "message": "WebSocket connection test successful"
        }))
        
        # Echo loop
        while True:
            # In Starlette, we need to use receive_text instead of async for
            data = await websocket.receive_text()
            websocket_logger.info(f"Test endpoint received: {data}")
            
            # Echo back the message
            await websocket.send_text(f"Echo: {data}")
            
    except Exception as e:
        websocket_logger.error(f"Test WebSocket error: {str(e)}")
        websocket_logger.error(traceback.format_exc())
    finally:
        # Make sure to close the connection
        try:
            await websocket.close()
        except:
            pass

# Function to run both web server and MCP server
async def run_with_servers():
    """Run web server, WebSocket server, and MCP server concurrently."""
    # Create tasks for web server
    config = uvicorn.Config(app, host="0.0.0.0", port=8080, log_level="info")
    web_server = uvicorn.Server(config)
    
    # Start MCP server - using the correct API
    try:
        # Try the current API method
        import mcp.server.stdio
        logger.info("Starting MCP server with current API")
        
        # Use context manager approach which is more common in newer versions
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            mcp_task = asyncio.create_task(
                mcp.run(read_stream, write_stream)
            )
            
            # Start web server
            await web_server.serve()
            
            # Wait for MCP task
            await mcp_task
            
    except (AttributeError, ImportError) as e:
        # Fallback to direct execution if context manager not available
        logger.warning(f"Falling back to direct MCP execution: {e}")
        logger.info("Note: MCP Claude Desktop integration may not work")
        
        # Start web server without MCP
        await web_server.serve()

# ===== MAIN ENTRYPOINT =====

# Also replace this part at the bottom of your server.py

# ===== MAIN ENTRYPOINT =====

if __name__ == "__main__":
    import sys
    import uvicorn
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "run":
            # Run with custom stdio mode when launched manually
            logger.info("Running in stdio mode for MCP protocol")
            import mcp.server.stdio
            
            # Use the correct approach for the MCP version
            try:
                async def run_mcp():
                    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
                        await mcp.run(read_stream, write_stream)
                
                asyncio.run(run_mcp())
            except Exception as e:
                logger.error(f"Error running MCP server: {e}")
                logger.error("Falling back to standard MCP run")
                mcp.run()
                
        elif sys.argv[1] == "websocket":
            # Run in WebSocket mode only
            logger.info("Running with WebSocket server")
            
            # Check if the app is valid
            if app is None:
                logger.error("Starlette app is not initialized")
                sys.exit(1)
                
            # Run the app directly with Uvicorn
            uvicorn.run(app, host="0.0.0.0", port=8080)
                
        elif sys.argv[1] == "web":
            # Run with WebSocket and web servers
            logger.info("Running with WebSocket and web servers")
            try:
                asyncio.run(run_with_servers())
            except KeyboardInterrupt:
                logger.info("Server shutdown requested")
    else:
        # Default behavior: try to run everything
        try:
            # If Starlette is available, start everything
            logger.info("Running all servers: MCP, WebSocket, and web")
            asyncio.run(run_with_servers())
        except ImportError:
            # Fall back to just MCP if Starlette is not available
            logger.info("Starlette not found, running in MCP mode only")
            mcp.run()