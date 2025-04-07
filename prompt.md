Can you run a simulation of an Antarctic ecosystem with 1000 leopard seals (predators) and 9000 emperor penguins (prey) over 50 steps? I'm interested in understanding how the predator-prey dynamics evolve in this harsh environment, especially considering the limited food resources. Please analyze the population trends and explain any important ecological observations.

I'd also like to add a 'Grow' substep to the simulation that allows the grass to regrow over time. When a grass patch has been eaten by prey, its growth stage should be set to 0 and its growth countdown timer should start. Each step of the simulation should decrease this countdown, and when it reaches zero, the growth stage should be set back to 1, making it available for consumption again. This substep should use the GrowGrassVmap generator and take grass_growth and growth_countdown as input variables, with the same variables as outputs.

Please analyze the population trends and explain any important ecological observations.

  '3':
    name: 'Grow'
    description: 'Grow Grass'
    active_agents:
      - 'prey'
    observation:
      prey: null
    policy:
      prey: null
    transition:
      grow_grass:
        generator: 'GrowGrassVmap'
        arguments: null
        input_variables:
          grass_growth: 'objects/grass/growth_stage'
          growth_countdown: 'objects/grass/growth_countdown'
        output_variables:
          - grass_growth
          - growth_countdown


We now want to achieve the below following tasks:
1) Connect MCP Server to Claude Desktop to run it.
2) Right now with our code for any prompt like hi, how are you doing, I am a biologist, etc the model is running the simulation instead knowing when to run the simulation which is very important. Because as we connect to our Claude desktop we need to be sure when user says to run the simulation the simulation should run and when he adds the keyword "grow" then only that that substep needs to be added. 
3) We need to come with the perfect demo for users to see, get attached to the mcp server of the agent framework and run it themselves. I have an idea but I want you to refine it further:
The user defines he/ she is a biologist working on understanding the habitat, animal situation in Antarctica. Then uses Claude web search to get roughly the idea which animals are popular as prey, predator, environment and get to know more about the Antarctica about the animal/habitat and environment. Using that information the user asks the model to run the simulation for a predator prey model in Antarctica by defining the predator, prey, etc and also telling the model to add a growth substep and come up with the simulation logs, visualisation, final result/ analysis. For each of these interaction by the user create the perfect , fun, human like interactive prompts which as a user I would ask. Also as we are using Claude desktop as the UI to connect to our local mcp server we want the simulation shown there while the user writes prompts on Claude desktop.