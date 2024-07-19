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
> first

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

## Execution
### Trainig
```
python scripts/train.py --config # Resume from config
python scripts/train.py --checkpoint # Resume from checkpoint
```

## Evaluating
There are multiple ways of evaluating. The simplest is
```
python scripts/test_policy.py --checkpoint 
```
Further options are `gui`: whether to use gui

If you want to evaluate about more runs with tracking, run
```
python scripts/kaggle.py
```
and set the variable `n_runs`, `config`, `checkpoint` accordingly