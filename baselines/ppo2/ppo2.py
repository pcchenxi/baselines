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

def learn(*, policy, env, nsteps, total_timesteps, ent_coef, lr,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
            save_interval=0):
    global min_return, max_return

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
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
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef, lr = lr(1),
                    max_grad_norm=max_grad_norm, need_summary = need_summary)

    model = make_model('model', need_summary = True)

    start_update = 0

    # load training policy
    checkdir = osp.join('../model', 'checkpoints')    
    # checkdir = osp.join('../model', 'fun_models')
    mocel_path = osp.join(checkdir, '%.5i'%(start_update))

    if start_update != 0:
        model.load(mocel_path)

    # model.load_value(checkdir+'/00088')

    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)

    nupdates = total_timesteps//nbatch

    max_return = [200, 120, 0, 0, 0]

    for update in range(start_update+1, nupdates+1):
        assert nbatch % nminibatches == 0
        # nbatch_train = nbatch // nminibatches
        nbatch_train = nminibatches    
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        lrnow = frac*lr(frac)
        cliprangenow = cliprange(frac)

        lrnow = np.clip(lrnow, 1e-4, 1)
        cliprangenow = np.clip(cliprangenow, 0.1, 1)

        print(lrnow, cliprangenow)

        # 1: sample batch using base policy
        obs, returns, masks, actions, values, rewards, neglogpacs, states, epinfos, ret = runner.run(nsteps * 4, is_test=False) #pylint: disable=E0632
        advs_ori = returns - values

        # mean_advs_ori = np.mean(advs_ori, axis = 0)

        base_mean_returns = np.mean(returns, axis = 0)

        # 2: update base value net
        mblossvals = nimibatch_update(   nbatch, noptepochs, nbatch_train,
                        obs, returns, masks, actions, values, advs_ori, neglogpacs,
                        lrnow, cliprangenow, states, nsteps, model, update_type = 'value')        
        # 3: get base policy params
        params_base = model.get_current_params()

        # 4: for loop:
        obs_p_buf, returns_p_buf, masks_p_buf, actions_p_buf, rewards_p_buf, values_p_buf, neglogpacs_p_buf, states_p_buf = [], [], [], [], [], [], [], []
        params_p_buf = []
        advs_p_buf = []

        for i in range(4): 
            print('--------------------------------', i)
            #   4.1: update base policy to a random direction

            # random_w = np.random.rand(REWARD_NUM)
            if i == 0:
                random_w = np.array([1, 0, 0, 0, 0])
            elif i == 1:
                random_w = np.array([0, 1, 0, 0, 0])
            elif i == 2:
                random_w = np.array([0, 0, 1, 0, 0])
            elif i == 3:
                random_w = np.array([0, 0, 0, 1, 0])

            print(' 4.1 update base policy', random_w)
            # print(np.array_str(mean_advs_ori, precision=3, suppress_small=True))

            advs = compute_advs(advs_ori, random_w)
            mblossvals = nimibatch_update(   nbatch, noptepochs, nbatch_train,
                        obs, returns, masks, actions, values, advs, neglogpacs,
                        lrnow, cliprangenow, states, nsteps, model, update_type = 'actor')       

            #   4.2: sample batch
            print(' 4.2 sample batch using updated policy')
            obs_p, returns_p, masks_p, actions_p, values_p, rewards_p, neglogpacs_p, states_p, epinfos_p, _ = runner.run(nsteps, is_test=False) #pylint: disable=E0632
            advs_p_ori = returns_p - values_p
            advs_p = compute_advs(advs_p_ori, np.array([1,1,1,1,1]))
            # print('adv mean and std', advs_p.mean(), advs_p.std())

            mean_returns = np.mean(returns_p, axis = 0)
            return_diff = mean_returns - base_mean_returns
            print(np.array_str(return_diff, precision=3, suppress_small=True))

            #   4.3: save updated policy and batch
            print(' 4.3 save updated policy and batch')
            if obs_p_buf == []:
                obs_p_buf, returns_p_buf, masks_p_buf, actions_p_buf, rewards_p_buf, values_p_buf, neglogpacs_p_buf, states_p_buf, epinfos_p_buf = obs_p, returns_p, masks_p, actions_p, rewards_p, values_p, neglogpacs_p, states_p, epinfos_p
                advs_p_buf = advs_p
                # params_p_buf = model.get_current_params()
            else:
                obs_p_buf = np.concatenate((obs_p_buf, obs_p), axis = 0)
                returns_p_buf = np.concatenate((returns_p_buf, returns_p), axis = 0)
                masks_p_buf = np.concatenate((masks_p_buf, masks_p), axis = 0)
                actions_p_buf = np.concatenate((actions_p_buf, actions_p), axis = 0)
                rewards_p_buf = np.concatenate((rewards_p_buf, rewards_p), axis = 0)
                neglogpacs_p_buf = np.concatenate((neglogpacs_p_buf, neglogpacs_p), axis = 0)
                # states_p_buf = np.concatenate((states_p_buf, states_p), axis = 0)
                advs_p_buf = np.concatenate((advs_p_buf, advs_p), axis = 0)
                values_p_buf = np.concatenate((values_p_buf, values_p), axis = 0)
                # params_p_buf = np.concatenate((params_p_buf, model.get_current_params()), axis = 0)
                
            #   4.4: restore base policy params
            print(' 4.4 restore base policy params', rewards_p.shape)
            model.replace_params(params_base)

        neglogpacs_p_buf = model.get_neglogpac(obs_p_buf, actions_p_buf)
        # neglogpacs

        mblossvals = nimibatch_update(   nbatch, noptepochs, nbatch_train,
                    obs_p_buf, returns_p_buf, masks_p_buf, actions_p_buf, values_p_buf, advs_p_buf, neglogpacs_p_buf,
                    lrnow, cliprangenow, states, nsteps, model, update_type = 'actor')    

        print('------------------- updated ------------------------------')

        # 5: evaluate each policy
        # 6: compute mata loss
        # 7: update base policy

        # # lrnow = lr
        # # cliprangenow = cliprange
        # obs, returns, masks, actions, values, rewards, neglogpacs, states, epinfos, ret = runner.run(is_test=False) #pylint: disable=E0632
        # # pi_mean, pi_std = model.get_actions_dist(obs)

        # advs_ori = returns - values
        # # advs = np.sum(advs_ori, axis=1)
        # # print(advs.shape)
        # advs = compute_advs(max_return, values, advs_ori, [], [])

        # mblossvals = nimibatch_update(   nbatch, noptepochs, nbatch_train,
        #                 obs, returns, masks, actions, values, advs, neglogpacs,
        #                 lrnow, cliprangenow, states, nsteps, model, update_type = 'all')
        display_updated_result( mblossvals, update, log_interval, nsteps, nbatch, 
                            rewards, returns, advs_ori, epinfos, model, logger)         

        if save_interval and (update % save_interval == 0 or update == 1 or update == nupdates) and logger.get_dir():
            #checkdir = osp.join(logger.get_dir(), 'checkpoints')
            checkdir = osp.join('../model', 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, '%.5i'%(update))
            savepath_base = osp.join(checkdir, '%.5i'%(0))
            print('Saving to', savepath, savepath_base)
            # np.save('../model/checkpoints/minmax_return_'+str(update), [min_return, max_return])
            # np.save('../model/checkpoints/minmax_return_'+str(00000), [min_return, max_return])
            model.save(savepath, savepath_base)

        first = False
    env.close()

