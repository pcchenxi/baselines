#!/usr/bin/env python3
import argparse
from baselines.common.cmd_util import mujoco_arg_parser
from baselines import bench, logger
from OpenGL import GL

def train(env_id, num_timesteps, seed):
    from baselines.common import set_global_seeds
    from baselines.common.vec_env.vec_normalize import VecNormalize
    from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
    from baselines.ppo2 import ppo2_test
    from baselines.ppo2.policies import MlpPolicy
    import gym, roboschool
    import tensorflow as tf
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()
    def make_env():
        #env = gym.make(env_id)
        env = gym.make("RoboschoolHalfCheetah-v1")
        env = bench.Monitor(env, logger.get_dir(), allow_early_resets=True, render = True)
        return env

    envs = []
    for _ in range(ncpu):
        envs.append(make_env)
    env = SubprocVecEnv(envs)
    #env = VecNormalize(env)

    set_global_seeds(seed)
    policy = MlpPolicy
    ppo2_test.learn(policy=policy, env=env, nsteps=1000, nminibatches=1000,
        lam=0.95, gamma=[0.99, 0.97, 0.97, 0.95, 0.95], noptepochs=15, log_interval=1,
        ent_coef=0.0,
        lr=3e-4,
        cliprange=0.2,
        #total_timesteps=num_timesteps,
        total_timesteps = 5e+7,
        save_interval = 5)


def main():
    args = mujoco_arg_parser().parse_args()
    logger.configure()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)


if __name__ == '__main__':
    main()
