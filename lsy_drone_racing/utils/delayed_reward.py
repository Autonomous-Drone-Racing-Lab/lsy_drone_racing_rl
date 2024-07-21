"""Delayes Reward class to delay rewards given to agent."""

class DelayedReward:
    """Reward to be given delayed by timesteps.

    Class to attach rewards to that should only be given after some time. I.e.
    the drone should only get reward for passing a gate after it has passed the
    gate and not immediately crashed
    """
    def __init__(self):
        """Create Delayed Reward object."""
        self.reward = None
        self.delay = None

    def add_reward(self, reward: float, delay: int):
        """Add a reward to be given after a delay.

        Args:
            reward: The reward to be given
            delay: The delay after which the reward should be given
        """
        if self.reward is not None or self.delay is not None:
            print("Warning: Overwriting existing reward or delay")
            print(f"Old reward: {self.reward}, old delay: {self.delay}")

        self.reward = reward
        self.delay = delay
    
    def step(self):
        """Decrease the delay by one."""
        if self.delay is not None:
            assert self.delay > 0, "Delay is already 0"
            self.delay -= 1
            
    
    def reset(self):
        """Reset the reward and delay."""
        self.reward = None
        self.delay = None
    
    def get_value(self, flush: bool=False) -> float:
        """Get the reward if it is ready.

        Args:
            flush: If True, the reward is given immediately
        """
        if self.reward is None or self.delay is None:
            return 0
        else:
            if flush or self.delay == 0:
                rew_to_return = self.reward
                self.reset()
                return rew_to_return
        
        return 0