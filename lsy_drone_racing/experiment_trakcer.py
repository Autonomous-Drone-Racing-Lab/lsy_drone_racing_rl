import numpy as np
import matplotlib.pyplot as plt
import pickle

class Experiment:
    """
    A class representing an experiment.

    Attributes:
        drone_obs (list): A list of drone observations.
        time_step (list): A list of time steps.

    Methods:
        from_dict(exp): Creates an Experiment object from a dictionary.
        add_drone_obs(drone_obs): Adds a drone observation to the list.
        add_time_step(time): Adds a time step to the list.
        get_drone_pos(): Returns the drone positions as a numpy array.
    """

    def __init__(self):
        self.drone_obs = []
        self.time_step = []

    @staticmethod
    def from_dict(exp):
        """
        Creates an Experiment object from a dictionary.

        Args:
            exp (dict): A dictionary containing the experiment data.

        Returns:
            Experiment: An Experiment object.
        """
        experiment = Experiment()
        experiment.drone_obs = exp["drone_obs"]
        experiment.time_step = exp["time_step"]
        return experiment

    def add_drone_obs(self, drone_obs: np.ndarray):
        """
        Adds a drone observation to the list.

        Args:
            drone_obs (numpy.ndarray): The drone observation to add.
        """
        self.drone_obs.append(drone_obs)
    
    def add_time_step(self, time):
        """
        Adds a time step to the list.

        Args:
            time: The time step to add.
        """
        self.time_step.append(time)
    
    def get_drone_pos(self):
        """
        Returns the drone positions as a numpy array.

        Returns:
            numpy.ndarray: The drone positions.
        """
        return np.array(self.drone_obs)

class ExperimentTracker:
    """
    A class for tracking and saving experiments.

    Attributes:
        experiments (list): A list of Experiment objects.

    Methods:
        add_experiment: Adds a new Experiment object to the experiments list.
        add_drone_obs: Adds drone observations and time step to the current experiment.
        save_experiment: Saves the experiments to a file using pickle.
    """

    def __init__(self):
        self.experiments = []
    
    def add_experiment(self):
        """
        Adds a new Experiment object to the experiments list.
        """
        self.experiments.append(Experiment())

    def add_drone_obs(self, drone_obs: np.ndarray, time_step):
        """
        Adds drone observations and time step to the current experiment.

        Args:
            drone_obs (np.ndarray): The drone observations.
            time_step: The time step of the observations.
        """
        self.experiments[-1].add_drone_obs(drone_obs)
        self.experiments[-1].add_time_step(time_step)

    def save_experiment(self, path):
        """
        Saves the experiments to a file using pickle.

        Args:
            path: The path to save the experiments file.
        """
        experiments = []
        for experiment in self.experiments:
            experiments.append({
                "drone_obs": experiment.drone_obs,
                "time_step": experiment.time_step
            })
        with open(path, 'wb') as f:
            pickle.dump(experiments, f)
        
