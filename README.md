# Autonomous Drone Racing Project Course

## Installation

To run the LSY Autonomous Drone Racing project, you will need 2 additional repositories:
- [safe-control-gym](https://github.com/Autonomous-Drone-Racing-Lab/safe-control-gym.git) - `lsy_drone_racing` branch: The drone simulator and gym environments. **Important, this repo is different from the original repo, as some changes to the code were conducted**
- [pycffirmware](https://github.com/utiasDSL/pycffirmware) - `main` branch: A simulator for the on-board controller response of the drones we are using to accurately model their behavior


### Create pyhthon environment
```bash
conda create -n drone python=3.8
conda activate drone
```

> **Note:** It is important you stick with **Python 3.8**. Yes, it is outdated. Yes, we'd also like to upgrade. However, there are serious issues beyond our control when deploying the code on the real drones with any other version.

Next, download the `safe-control-gym` and `pycffirmware` repositories and install them. Make sure you have your conda/mamba environment active!

```bash
cd ~/repos
git clone https://github.com/Autonomous-Drone-Racing-Lab/safe-control-gym.git
cd safe-control-gym
pip install .
```

> **Note:** If you receive an error installing safe-control-gym related to gym==0.21.0, run
> ```bash
>    pip install setuptools==65.5.0 pip==21 wheel==0.38.4
> ```

```bash
cd ~/repos
git clone https://github.com/utiasDSL/pycffirmware.git
cd pycffirmware
git submodule update --init --recursive
sudo apt update
sudo apt install build-essential
conda install swig
./wrapper/build_linux.sh
```
Finally install stable baselines 3 for training the RL agents
```
pip install stable-baselines3
```

> Important: Do not install the current repo with `pip install .` as this leads to issues with some scripts

## Execution

### Configuration
All configuration is done via a config file as shown in `config`. Default values are not supported and all values must be set. If unsure just use the values as provided here. The config files defined both the environment and the RL agent. For adapting the RL agent, epsecially the `rl_config` fields are important.


### Trainig
Training is done over the `scripts/train.py` file with options for either starting a new run based on the configuration or resume an existing run. For new run you must provide the path to the config file, e.g. `config/baseline.yaml`. REsuming you provide the path to a checkpoint, the mathching config will automatically be loaded from the folder. During training the agent will be evaluated every 50.000 steps. Per default this happens in a level3 environment. If you want to utilize a different environment you must manually change the code.

To summarize:
```
python scripts/train.py --config # Resume from config
python scripts/train.py --checkpoint # Resume from checkpoint
```

## Evaluating
There are multiple ways of evaluating the agent. 

The simplest to just test an agen without tracking is to test it from a checkpoint. For this utilize the function
```
python scripts/test_policy.py --checkpoint <path to checkpoint> --gui <whether to use gui>
```


If you want to evaluate about more runs and track its result you can use.
```
python scripts/kaggle.py
```
This function does not provide as nice of a command line interface, rather you must manually set the variables `n_runs`, `config`, `checkpoint` according to your desire. Config should typically be any of the eval configs [level0, ..., level3]. These configs do not provide and RL configuration. Rather the RL configuration is loaded form the `config.yaml` file next to the checkpoint. Therefore you must make sure the original folder structure is preserved.