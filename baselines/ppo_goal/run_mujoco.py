#!/usr/bin/env python3
import argparse
from baselines.common.cmd_util import mujoco_arg_parser
from baselines import bench, logger
from OpenGL import GL

def train(env_id, num_timesteps, seed):
    from baselines.common import set_global_seeds
    from baselines.common.vec_env.vec_normalize import VecNormalize
    from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
    # from baselines.ppo2 import ppo_mpi_gp as ppo2
    from baselines.ppo_goal import ppo_mpi_morl as ppo2
    # from baselines.ppo2 import ppo_mpi_fine_tune as ppo2
    # from baselines.ppo2 import ppo_policy_plot as ppo2
    # from baselines.ppo_goal import ppo_mpi_normal as ppo2
    from baselines.ppo_goal.policies import MlpPolicy_Goal as MlpPolicy
    import gym, roboschool
    import gym_hockeypuck
    import tensorflow as tf
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()
    def make_env():
        #env = gym.make(env_id)
        # env = gym.make("hockeypuck-v0")
        env = gym.make("FetchPickAndPlace-v1")
        # env = bench.Monitor(env, logger.get_dir(), allow_early_resets=True)
        return env

    envs = []
    for _ in range(ncpu):
        envs.append(make_env)
    env = gym.make("FetchPickAndPlace-v1") #SubprocVecEnv(envs)
    # env = VecNormalize(env)
    # set_global_seeds(seed)
    policy = MlpPolicy

    nsteps = 256/ncpu

    ppo2.learn(policy=policy, env=env, nsteps=int(nsteps), nminibatches=int(256),
        lam=0.99, gamma=0.99, noptepochs=5, log_interval=1,
        ent_coef=0.000, # 0.003,
        lr= 3e-4,
        cliprange=0.2,
        #total_timesteps=num_timesteps,
        total_timesteps = 10e+7,
        save_interval = 1)


def main():
    args = mujoco_arg_parser().parse_args()
    logger.configure()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)


if __name__ == '__main__':
    main()
