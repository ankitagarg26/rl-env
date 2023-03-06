import gymnasium as gym


class MovetoFiveDirectionsWrapper(gym.core.Wrapper):
    """
    A wrapper to remove unused actions.
    """

    def __init__(self, env):
        super(MovetoFiveDirectionsWrapper, self).__init__(env)
       
    def step(self, action):
        if action == 4:
            return self.env.step(5)
        else:
            return self.env.step(action)