class DelayedReward:
    """
    Class to attach rewards to that should only be given after some time. I.e. 
    the drone should only get reward for passing a gate after it has passed the
    gate and not immediately crashed
    """
    def __init__(self):
        self.reward = None
        self.delay = None

    def add_reward(self, reward, delay):
        """
        Add a reward to be given after a delay
        """
        assert self.reward is None, "Reward already set"
        assert self.delay is None, "Delay already set"
    
        self.reward = reward
        self.delay = delay
    
    def step(self):
        """
        Decrease the delay by one
        """
        if self.delay is not None:
            assert self.delay > 0, "Delay is already 0"
            self.delay -= 1
            
    
    def reset(self):
        """
        Reset the reward and delay
        """
        self.reward = None
        self.delay = None
    
    def get_value(self, flush=False):
        """
        Get the reward if it is ready
        """
        if self.reward is None or self.delay is None:
            return 0
        else:
            if flush or self.delay == 0:
                rew_to_return = self.reward
                self.reset()
                return rew_to_return
        
        return 0