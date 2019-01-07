import os
import time
import joblib
import math
import numpy as np
import os.path as osp
import tensorflow as tf
from baselines import logger
from collections import deque
from baselines.ppo_goal.common_functions import *
import matplotlib.pyplot as plt
from mpi4py import MPI
import datetime
import scipy


def learn(*, policy, env, nsteps, total_timesteps, ent_coef, lr,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
            save_interval=0):

    first = True

    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    nenvs = 1 #env.num_envs
    ob_spaces = env.observation_space.spaces.items()
    ac_space = env.action_space

    index = 0
    for a in ob_spaces:
        if index == 2:
            ob_space = a[1]
        else:
            goal_space = a[1]
        index += 1

    nbatch = nenvs * nsteps
    # nbatch_train = nbatch // nminibatches

    nbatch_train = nminibatches
    print('nbatch', nbatch, nenvs, nsteps, nbatch % nminibatches, nbatch_train)
    make_model = lambda model_name, need_summary : Model(model_name=model_name, policy=policy, ob_space=ob_space, ac_space=ac_space, goal_space=goal_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef, lr = lr,
                    max_grad_norm=max_grad_norm, need_summary = need_summary)

    model = make_model('model', need_summary = True)

    start_update = 0

    # load training policy
    checkdir = osp.join('../model', 'checkpoints')    
    # checkdir = osp.join('../model', 'fun_models')
    mocel_path = osp.join(checkdir, str(start_update))

    if start_update != 0:
        model.load(mocel_path)
        # pf_sets = np.load('../model/checkpoints/pf.npy').tolist()
        # pf_sets_convex = np.load('../model/checkpoints/pf_convex.npy').tolist()

    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)

    nupdates = total_timesteps//nbatch

    for update in range(start_update+1, nupdates+1):
        runner.init_task_pool(1000)
        obs, returns, dones, actions, values, advs_ori, rewards, neglogpacs, epinfos, ret = runner.run(int(nsteps), is_test=False) #pylint: disable=E0632
        # advs = advs_ori[:, -1] #+ advs_ori[:, 1]*2
        advs = np.asarray(advs_ori)

        obs_gather = MPI.COMM_WORLD.gather(obs)
        returns_gather = MPI.COMM_WORLD.gather(returns)
        actions_gather = MPI.COMM_WORLD.gather(actions)
        advs_gather = MPI.COMM_WORLD.gather(advs)
        values_gather = MPI.COMM_WORLD.gather(values)
        neglogpacs_gather = MPI.COMM_WORLD.gather(neglogpacs)

        if comm_rank == 0:
            all_obs, all_a, all_ret, all_adv, all_value, all_neglogpacs = [], [], [], [], [], []
            for obs, actions, returns, advs, values, neglogpacs in zip(obs_gather, actions_gather, returns_gather, advs_gather, values_gather, neglogpacs_gather):
                all_obs, all_a, all_ret, all_adv, all_value, all_neglogpacs = stack_values([all_obs, all_a, all_ret, all_adv, all_value, all_neglogpacs], [obs, actions, returns, advs, values, neglogpacs])

            print(all_obs.shape, all_ret.shape)

            print('advs mean', advs.mean(), advs.max(), advs.min())
            print('ret mean', returns.mean(), returns.max(), returns.min())
            print(values.mean(), rewards.mean())

            # mean_returns = np.mean(returns, axis = 0)
            # mean_values = np.mean(values, axis = 0)
            # mean_rewards = np.mean(advs_ori, axis = 0)

            # print(obs.shape, advs.shape, returns.shape)
            # print(update, comm_rank, 'ret    ', np.array_str(mean_returns, precision=3, suppress_small=True))
            # print(update, comm_rank, 'val    ', np.array_str(mean_values, precision=3, suppress_small=True))
            # print(update, comm_rank, 'adv    ', np.array_str(mean_rewards, precision=3, suppress_small=True))
            # print(values_ret[:,-1].mean())

            lr_p, cr_p = 0.0003, 0.2

            mblossvals = nimibatch_update(  nbatch, noptepochs, nbatch_train,
                                            obs, returns, actions, values, advs, neglogpacs,
                                            lr_p, cr_p, nsteps, model, update_type = 'all')   
            print(mblossvals[0], mblossvals[1], mblossvals[2])
            # display_updated_result( mblossvals, update, log_interval, nsteps, nbatch, 
            #                     rewards, returns, values, advs_ori, epinfos, model, logger)    


            print('\n') 

            if save_interval and (update % save_interval == 0 or update == 1 or update == nupdates) and logger.get_dir():
                save_model(model, update)
                save_model(model, 0)

            first = False
    env.close()

