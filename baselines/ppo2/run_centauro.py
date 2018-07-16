#!/usr/bin/env python3
import argparse, time
from baselines.common.cmd_util import mujoco_arg_parser
from baselines import bench, logger
from OpenGL import GL
import numpy as np


def train(env_id, num_timesteps, seed):
    from baselines.common import set_global_seeds
    # from baselines.common.vec_env.vec_normalize import VecNormalize
    from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
    from baselines.ppo2 import ppo2
    from baselines.ppo2.policies import MlpPolicy
    import gym, roboschool
    import tensorflow as tf
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    ncpu = 8
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()

    def make_env(env_id):
        #env = gym.make(env_id)
        env = gym.make("centauro_vrep-v0")
        env.seed(20000 + env_id)
        env = bench.Monitor(env, logger.get_dir(), allow_early_resets=True)
        print('environment id in make env', env_id)
        return env

    envs = []
    for _ in range(ncpu):
        envs.append(make_env)
    env = SubprocVecEnv(envs)
    #env = VecNormalize(env)

    set_global_seeds(seed)
    policy = MlpPolicy

    nsteps = int(512*16/ncpu)
    ppo2.learn(policy=policy, env=env, nsteps=nsteps, nminibatches=2048,
        lam=0.98, gamma=np.asarray([0.9, 0.9, 0.98, 0.9]), noptepochs=10, log_interval=1,
        ent_coef=0.0,
        lr=3e-4,
        cliprange=0.2,
        #total_timesteps=num_timesteps,
        total_timesteps = 50e+6,
        save_interval = 10)


def main():
    args = mujoco_arg_parser().parse_args()
    logger.configure()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)


if __name__ == '__main__':
    main()
