"""
The self-defined environments:
"""

from minigrid.core.mission import MissionSpace
from minigrid.core.grid import Grid
from minigrid.core.world_object import Door, Goal, Key

from minigrid.minigrid_env import MiniGridEnv
from gymnasium.envs.registration import register


class ComplexEnv(MiniGridEnv):
    """
    The class to define some more complex environments.
    """

    def __init__(self, mission_space, width, height, max_steps):
        
        super().__init__(mission_space = mission_space, width = width, height = height, max_steps = max_steps)
        

    def _gen_grid(self, width, height):
        # to create an empty grid
        self.grid = Grid(width, height)
        
        # to create the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # to place the goal
        self.goal_loc = (width - 2, height - 2)
        self.put_obj(Goal(), width - 2, height - 2)

        
        splitIdx = self._rand_int(2, width - 2)
            
        # to place the agent at its initial location:
        self.agent_init_loc = self.place_agent(size=(splitIdx, height))
        

        # to place the walls
        self.wall_locs = []
        self.grid.vert_wall(splitIdx, 0)
        for j in range(0, height):
            self.wall_locs.append((splitIdx, j))

        # to place the door
        self.door_locs = []
        doorIdx = self._rand_int(1, width - 2)
        self.door_locs.append((splitIdx, doorIdx))
        self.put_obj(Door("yellow", is_locked=True), splitIdx, doorIdx)

        # to place the key
        self.key_locs = []
        key_pos = self.place_obj(obj=Key("yellow"), top=(0, 0), size=(splitIdx, height))
        self.key_locs.append((key_pos[0], key_pos[1]))

        self.mission = "Get to the goal"


class ExpEnvV2(ComplexEnv):
    
    def __init__(self):
        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(mission_space, width=10, height=10, max_steps=1000)
        
    @staticmethod
    def _gen_mission():
        return "pick up the key, open the door and go to the goal location"



register(
    id='MiniGrid-Exp-V2-10x10',
    entry_point='envs.envs:ExpEnvV2'
)