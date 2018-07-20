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

def stack_values(buffers, values):
    if buffers[0] == []:
        for i in range(len(buffers)):
            buffers[i] = np.asarray(values[i])
    else:
        for i in range(len(buffers)):
            if buffers[i].all() != None:
                buffers[i] = np.concatenate((buffers[i], np.asarray(values[i])), axis = 0)  

    return buffers

def learn(*, policy, env, nsteps, total_timesteps, ent_coef, lr,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
            save_interval=0):
    global min_return, max_return

    first = True

    # if isinstance(lr, float): lr = constfn(lr)
    # else: assert callable(lr)
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
    mocel_path = osp.join(checkdir, '%.5i'%(start_update))

    if start_update != 0:
        model.load(mocel_path)

    # model.load_value(checkdir+'/00088')

    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)

    nupdates = total_timesteps//nbatch

    max_return = [200, 120, 0, 0, 0]

    for update in range(start_update+1, nupdates+1):
        # assert nbatch % nminibatches == 0
        # nbatch_train = nbatch // nminibatches
        nbatch_train = nminibatches    
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        # lrnow = frac*lr(frac)
        lrnow = lr
        cliprangenow = cliprange(frac)

        # lrnow = np.clip(lrnow, 1e-4, 1)
        # cliprangenow = np.clip(cliprangenow, 0.1, 1)

        print('lr', lrnow, cliprangenow)


        # 1: sample batch using base policy
        obs, returns, masks, actions, values, rewards, neglogpacs, states, epinfos, ret = runner.run(nsteps, is_test=False) #pylint: disable=E0632
        advs_ori = returns - values
        print('base batch size', values.shape)
        # mean_advs_ori = np.mean(advs_ori, axis = 0)
        base_mean_returns = np.mean(returns, axis = 0)     

        # advs = compute_advs(advs_ori, np.asarray([1, -1, 1, 1, 1]))
        # # advs = compute_advs_singleobj(values, returns)
        # mblossvals = nimibatch_update(   nbatch, noptepochs, nbatch_train,
        #                 obs, returns, masks, actions, values, advs, neglogpacs,
        #                 lrnow, cliprangenow, states, nsteps, model, update_type = 'all')    


        # # 2: update base value net
        # mblossvals = nimibatch_update(   nbatch, noptepochs, nbatch_train,
        #                 obs, returns, masks, actions, values, advs_ori, neglogpacs,
        #                 lrnow, cliprangenow, states, nsteps, model, update_type = 'value')   



        # 3: get base policy params
        params_base = model.get_current_params()

        # 4: meta loop:
        obs_p_buf, returns_p_buf, masks_p_buf, actions_p_buf, rewards_p_buf, values_p_buf, neglogpacs_p_buf, states_p_buf, advs_p_buf = [], [], [], [], [], [], [], [], []
        obs_buf, returns_buf, masks_buf, actions_buf, rewards_buf, values_buf, neglogpacs_buf, states_buf, advs_buf = [], [], [], [], [], [], [], [], []

        buffers_p = [obs_p_buf, returns_p_buf, masks_p_buf, actions_p_buf, rewards_p_buf, values_p_buf, neglogpacs_p_buf, states_p_buf, advs_p_buf]
        buffers_base = [obs_buf, returns_buf, masks_buf, actions_buf, rewards_buf, values_buf, neglogpacs_buf, states_buf, advs_buf]

        params_p_buf = []

        vote_1, vote_2 = 0, 0

        for i in range(2): 
            print('--------------------------------', i)
            # # 1: sample batch using base policy
            # print('collect batch using base policy')
            # obs, returns, masks, actions, values, rewards, neglogpacs, states, epinfos, ret = runner.run(int(nsteps), is_test=False) #pylint: disable=E0632
            # advs_ori = returns - values
            # base_mean_returns = np.mean(returns, axis = 0)     
            # print('base batch size', values.shape)
            # #   4.1: update base policy to a random direction

            # random_w = np.random.rand(REWARD_NUM)
            print (vote_1, vote_2)

            if i%2 == 0:
                random_w_p = np.array([0, 1, 0, 0, 0])
                random_w = np.array([1, 1, 1, 1, 1])
            elif i%2 != 0:
                random_w_p = np.array([0, -1, 0, 0, 0])
                random_w = np.array([1, -1, 1, 1, 1])
            # if i > 10 and vote_1 < -vote_2:
            #     random_w = np.array([0, 1, 0, 1, 0])
            # if i > 10 and vote_1 > -vote_2:
            #     random_w = np.array([0, -1, 0, 1, 0])
            # elif i == 2:
            #     random_w = np.array([0, 0, 1, 0, 0])
            # elif i == 3:
            #     random_w = np.array([0, 0, 0, 1, 0])

            print(' 4.1 update base policy', random_w)
            # print(np.array_str(mean_advs_ori, precision=3, suppress_small=True))

            # lr_p = 0.001
            advs = compute_advs(advs_ori, random_w_p)

            mblossvals = nimibatch_update(   nbatch, noptepochs, nbatch_train,
                        obs, returns, masks, actions, values, advs, neglogpacs,
                        lrnow, cliprangenow*2, states, nsteps, model, update_type = 'actor')       

            if (update % save_interval == 0 or first) and i%2 == 0:
                checkdir = osp.join('../model', 'checkpoints')
                savepath_base = osp.join(checkdir, '%.5i'%(1))
                print('Saving to', savepath_base, savepath_base)
                model.save(savepath_base, savepath_base)
            if (update % save_interval == 0 or first) and i%2 != 0:
                checkdir = osp.join('../model', 'checkpoints')
                savepath_base = osp.join(checkdir, '%.5i'%(2))
                print('Saving to', savepath_base, savepath_base)
                model.save(savepath_base, savepath_base)

            #   4.2: sample batch
            print(' 4.2 sample batch using updated policy')
            obs_p, returns_p, masks_p, actions_p, values_p, rewards_p, neglogpacs_p, states_p, epinfos_p, _ = runner.run(int(nsteps), is_test=False) #pylint: disable=E0632

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # update value network using the new batchs to compute advs which make sense for all objectives
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            mblossvals = nimibatch_update(   nbatch, noptepochs, nbatch_train,
                        obs_p, returns_p, masks_p, actions_p, values_p, advs, neglogpacs,
                        lrnow, cliprangenow, states, nsteps, model, update_type = 'values')  
            values_p = model.get_values(obs_p)

            if i%2 == 0:
                for r_index in range(REWARD_NUM):
                    model.write_summary('meta_f/mean_ret'+str(r_index+1), np.mean(returns_p[:,r_index]), update)
            else:
                for r_index in range(REWARD_NUM):
                    model.write_summary('meta_b/mean_ret'+str(r_index+1), np.mean(returns_p[:,r_index]), update)

            advs_p_ori = returns_p - values_p
            advs_p_ori = advs_p_ori/2
            advs_p_ori[:,1] = advs_p_ori[:,1]*2

            advs_p = compute_advs(advs_p_ori, random_w)
            print('updated batch size', values_p.shape)

            # print('adv mean and std', advs_p.mean(), advs_p.std())

            mean_returns = np.mean(returns_p, axis = 0)
            return_diff = mean_returns - base_mean_returns

            if i%2 == 0:
                vote_1 += return_diff[1]
            if i%2 != 0:
                vote_2 += return_diff[1]
            print(np.array_str(return_diff, precision=3, suppress_small=True))

            #   4.3: save updated batch
            datas_p = [obs_p, returns_p, masks_p, actions_p, rewards_p, values_p, neglogpacs_p, states_p, advs_p]
            datas_base = [obs, returns, masks, actions, rewards, values, neglogpacs, states, advs]

            obs_p_buf, returns_p_buf, masks_p_buf, actions_p_buf, rewards_p_buf, values_p_buf, neglogpacs_p_buf, states_p_buf, advs_p_buf = stack_values(buffers_p, datas_p)
            obs_buf, returns_buf, masks_buf, actions_buf, rewards_buf, values_buf, neglogpacs_buf, states_buf, advs_buf = stack_values(buffers_base, datas_base)

            #   4.4: restore base policy params
            model.replace_params(params_base)

            # mean_a, std_a = model.get_actions_dist(obs)
            # print(mean_a)

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! re-compute old probs and values
        neglogpacs_p_buf = model.get_neglogpac(obs_p_buf, actions_p_buf)
        values_p_buf = model.get_values(obs_p_buf)

        # neglogpacs
        print(rewards_buf.shape, rewards_p_buf.shape)

        # update base value net
        # mblossvals = nimibatch_update(   nbatch, noptepochs, nbatch_train,
        #                 obs_buf, returns_buf, masks_buf, actions_buf, values_buf, advs_buf, neglogpacs_p_buf,
        #                 lrnow, cliprangenow, states, nsteps, model, update_type = 'value')   
        
        # update base actor net
        mblossvals = nimibatch_update(   nbatch, noptepochs, nbatch_train,
                    obs_p_buf, returns_p_buf, masks_p_buf, actions_p_buf, values_p_buf, advs_p_buf, neglogpacs_p_buf,
                    lrnow, cliprangenow, states, nsteps, model, update_type = 'all')    

        print('mean advs', np.mean(advs_p_buf, axis=0))
        print('------------------- updated ------------------------------')

        display_updated_result( mblossvals, update, log_interval, nsteps, nbatch, 
                            rewards_buf, returns_p_buf, advs_p_buf, epinfos, model, logger)         

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

