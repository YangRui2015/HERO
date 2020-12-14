# Multi-step HER
Multi-step HER supports multi-step HER and is revised from OpenAI baselines.

## Installation
- Install the requirements such as *tensorflow*, *mpi4py* and *mujoco_py* using pip, besides multi-world should be installed from this repo https://github.com/vitchyr/multiworld;

- Clone the repo and cd into it;

- Install baselines package
    ```bash
    pip install -e .
    ```


## Usage
Experiment environments: SawyerReachXYZEnv-v1, SawyerPushAndReachEnvEasy-v0, FetchReach-v1, FetchPush-v1, FetchSlide-v1, FetchPickAndPlace-v1, HandReach-v0,  HandManipulateBlockRotateXYZ-v0.

DDPG:
```bash
python -m  baselines.run  --env=FetchPush-v1 --num_epoch 50 --num_env 12 --noher True --log_path=~/logs/FetchPush_env12/ --save_path=~/ddpg/fetchpush/
```
HER:
```bash
python -m  baselines.run  --env=FetchPush-v1 --num_epoch 50 --num_env 12 --log_path=~/logs/FetchPush_env12/ --save_path=~/her/fetchpush/
```
vanilla MHER:
```bash
python -m  baselines.run  --env=FetchPush-v1 --num_epoch 50 --num_env 12  --n_step 2 --mode nstep --log_path=~/logs/FetchPush_env12_nstep_2/ --save_path=~/policies/nstepher/fetchpush/
```
MHER($\lambda$):
```bash
python -m  baselines.run  --env=FetchPush-v1 --num_epoch 50 --num_env 12  --n_step 2 --mode lambda --lamb 0.7 --log_path=~/logs/FetchPush_env12_nstep_2/ --save_path=~/policies/mher_lambda/fetchpush/
```
Model-based MHER:
```bash
python -m  baselines.run --env=FetchPush-v1 --num_epoch 50 --num_env 12  --n_step 2 --mode dynamic --alpha 0.5 --log_path=~/logs/FetchPush_env12_nstep_2/ --save_path=~/policies/mmher/fetchpush/
```
HER + Correction:
```bash
python -m  baselines.run  --env=FetchPush-v1 --num_epoch 50 --num_env 12  --n_step 2 --mode correct --cor_rate 1 --log_path=~/logs/FetchPush_env12_nstep_2/ --save_path=~/policies/nstepher_correction/fetchpush/
```
