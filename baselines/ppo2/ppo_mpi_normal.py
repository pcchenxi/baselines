import os
import time
import joblib
import math
import numpy as np
import os.path as osp
import tensorflow as tf
from baselines import logger
from collections import deque
from baselines.common import explained_variance
from baselines.ppo2.ratio_functions import *
from baselines.ppo2.common_functions import *
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from mpi4py import MPI
import datetime


def learn(*, policy, env, nsteps, total_timesteps, ent_coef, lr,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
            save_interval=0):

    first = True

    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch = nenvs * nsteps
    # nbatch_train = nbatch // nminibatches

    nbatch_train = nminibatches
    print('nbatch', nbatch, nenvs, nsteps, nbatch % nminibatches, nbatch_train)
    make_model = lambda model_name, need_summary : Model(model_name=model_name, policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
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

    weight = np.ones(REWARD_NUM)
    for update in range(start_update+1, nupdates+1):
        obs, returns, masks, actions, values, rewards, neglogpacs, states, epinfos, ret = runner.run(int(nsteps), is_test=False) #pylint: disable=E0632
        advs = returns - values
        advs = compute_advs(advs, weight)

        mean_returns = np.mean(returns, axis = 0)

        print(obs.shape, advs.shape)
        print(update, comm_rank, '    ', np.array_str(mean_returns, precision=3, suppress_small=True))
        lr_p, cr_p = 0.0003, 0.2
        mblossvals = nimibatch_update(  nbatch, noptepochs, nbatch_train,
                                        obs, returns, masks, actions, values, advs, neglogpacs,
                                        lr_p, cr_p, states, nsteps, model, update_type = 'all')   
        display_updated_result( mblossvals, update, log_interval, nsteps, nbatch, 
                            rewards, returns, advs, epinfos, model, logger)    


        print('\n') 

        if save_interval and (update % save_interval == 0 or update == 1 or update == nupdates) and logger.get_dir():
            save_model(model, update)
            save_model(model, 0)

        first = False
    env.close()

