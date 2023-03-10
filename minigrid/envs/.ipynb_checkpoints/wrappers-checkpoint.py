import gymnasium as gym


class SubGoalsIndicator(gym.core.Wrapper):
    """
    The wrapper to re-assign some reward signals for some key points (or sub-goals).
    """

    def __init__(self, env, self_defined_goals=None):
        super(SubGoalsIndicator, self).__init__(env)
        assert len(env.unwrapped.key_locs) == 1, "The wrapper is only applicable for the env with only one key."

        if self_defined_goals:
            # assert len(self_defined_goals) == 2, "Designed wrapper only for two additional sub-goals"
            self.self_defined_goals = self_defined_goals
        else:
            self.self_defined_goals = None

    def reset(self, **kwargs):
        obs = self.env.reset()

        # * set the flag to make sure the goal can be completed only once.
        self.unwrapped.key_picked = False
        self.unwrapped.door_passed = False

        return obs

    def step(self, action):
        obs, reward, done, terminate, info = self.env.step(action)

        info['key_picked'] = False
        info['door_passed'] = False

        env = self.unwrapped

        agent_loc = env.agent_pos
        agent_dir = env.agent_dir

        if not isinstance(agent_loc, tuple):
            agent_loc = tuple(env.agent_pos.tolist())

        key_loc = env.key_locs[0]
        door_loc = env.door_locs[0]

        if (action == 3 and (env.key_picked == False)):
            # * when the agent picks up the key the first time
            if (((agent_dir == 0) and ((agent_loc[0] + 1, agent_loc[1]) == key_loc)) or
            ((agent_dir == 2) and ((agent_loc[0] - 1, agent_loc[1]) == key_loc)) or 
            ((agent_dir == 1) and ((agent_loc[0], agent_loc[1] + 1) == key_loc)) or
            ((agent_dir == 3) and ((agent_loc[0], agent_loc[1] - 1) == key_loc))):
                info['key_picked'] = True
                env.key_picked = True

        if (action == 4 and (env.door_passed == False) and (env.key_picked == True)):
            # * when the agent pass the door for the first time
            if (((agent_dir == 0) and ((agent_loc[0] + 1, agent_loc[1]) == door_loc)) or
            ((agent_dir == 2) and ((agent_loc[0] - 1, agent_loc[1]) == door_loc)) or 
            ((agent_dir == 1) and ((agent_loc[0], agent_loc[1] + 1) == door_loc)) or
            ((agent_dir == 3) and ((agent_loc[0], agent_loc[1] - 1) == door_loc))):
                info['door_passed'] = True
                env.door_passed = True

        return obs, reward, done, terminate, info