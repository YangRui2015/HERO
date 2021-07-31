# HERO
Code for **HERO: Combining Hindsight and Goal-enhanced Prediction in Multi-goal Reinforcement Learning**.

## Installation
- Install the requirements such as *tensorflow*, *mpi4py*, *gym*, and *mujoco_py* using pip;

- Clone the repo and cd into it;

- Install baselines package
    ```bash
    pip install -e .
    ```


## Usage
Experiment environments: FetchReach-v1, FetchPush-v1, FetchSlide-v1, FetchPickAndPlace-v1.

DDPG:
```bash
python -m  baselines.run  --env=FetchPush-v1 --num_epoch 50 --num_env 12 --noher True --log_path=~/logs/FetchPush_env12/ --save_path=~/ddpg/fetchpush/
```
HER:
```bash
python -m  baselines.run  --env=FetchPush-v1 --num_epoch 50 --num_env 12 --log_path=~/logs/FetchPush_env12/ --save_path=~/her/fetchpush/
```
MBPO + HER:
```bash
python -m  baselines.run  --env=FetchPush-v1 --num_epoch 50 --num_env 12 --mode mbpo --n_step 5  --log_path=~/logs/FetchPush_env12/ --save_path=~/mbpo/fetchpush/
```

HERO:
```bash
python -m  baselines.run --env=FetchPush-v1 --num_epoch 50 --num_env 12  --n_step 2 --mode hero --alpha 0.4 --log_path=~/logs/FetchPush_env12_nstep_2/ --save_path=~/policies/mmher/fetchpush/
```

## Main Functions
The main functions of our algorithms are in /baselines/her/her_sampler.py. Names of our main function is *_sample_hero_transitions*.

