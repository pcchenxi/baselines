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

def update_pf_sets(pf_sets, candidate_set, value_loss):
    keep_weight = False 

    if pf_sets == []:
        pf_sets.append(candidate_set)
        keep_weight = True 
    else:
        add_candidate = False
        for i in range(len(pf_sets)):
            if np.all(candidate_set > np.asarray(pf_sets[i])): # pf_sets[i] is pareto dominated by the candidate, remove pf_sets[i] and add candidate to the pf_sets
                # pf_sets.pop(i)
                pf_sets[i] = candidate_set
                add_candidate = True
            elif np.any(candidate_set > np.asarray(pf_sets[i])): # candidate is a part of current pf
                add_candidate = True
        if add_candidate:
            pf_sets.append(candidate_set)
            keep_weight = True 
        else:
            keep_weight = False

    return keep_weight


def compute_weights(w_vs, pf_sets, w_returns, target_w):
    target_vs = w_vs
    for pf_set in pf_sets:
        pf_v = np.sum(np.multiply(pf_set, target_w))
        target_vs.append(pf_v)

    target_vs = np.asarray(target_vs)
    target_vs = target_vs - target_vs.min()
    target_vs = target_vs/target_vs.max()
    for i in range(len(w_returns)):
        for pf_set in pf_sets:
            if np.all(pf_set == w_returns[i]):
                target_vs[i] += 1
                break

    target_vs_normalized = target_vs/target_vs.max()
    target_vs_normalized = np.clip(target_vs_normalized, 0, 1)

    return target_vs_normalized[:len(w_returns)]

def learn(*, policy, env, nsteps, total_timesteps, ent_coef, lr,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
            save_interval=0):

    first = True
    pf_sets = []
    target_w = [1.0, 1.0, 1.0, 1.0, 1.0]
    best_target_v = 0

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

    start_update = 18

    # load training policy
    checkdir = osp.join('../model', 'checkpoints')    
    # checkdir = osp.join('../model', 'fun_models')
    mocel_path = osp.join(checkdir, str(start_update))

    if start_update != 0:
        model.load(mocel_path)
        # pf_sets = np.load('../model/checkpoints/pf.npy').tolist()

    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)

    nupdates = total_timesteps//nbatch

    for update in range(start_update+1, nupdates+1):
        # assert nbatch % nminibatches == 0
        # nbatch_train = nbatch // nminibatches
        nbatch_train = nminibatches    
        tstart = time.time()

        # save the base policy params
        # save_model(model, 0)
        params_all_base = model.get_current_params(params_type='all')
        params_actor_base = model.get_current_params(params_type='actor')
        params_value_base = model.get_current_params(params_type='value')

        all_obs, all_a, all_ret = [], [], []
        for i in range(40): 
            keep_weight = False
            print('--------------------------------', i, '--------------------------------')
            # 2: update base policy to a random direction
            random_w = (np.random.rand(REWARD_NUM) - 0.5)*2
            random_w[1] = abs(random_w[1])
            random_w[0] = abs(random_w[0])
            # if i == 0:
            #     random_w = [0, 1,  0,  0, 0]
            # elif i == 1:
            #     random_w = [0, -1,  0,  0, 0]

            num_g = 3
            v_loss_pre = 0
            for g in range(num_g):
                obs, returns, masks, actions, values, rewards, neglogpacs, states, epinfos, ret = runner.run(int(nsteps), is_test=False) #pylint: disable=E0632
                advs_ori = returns - values
                print('    base batch collected', values.shape, g)
                print('adv',advs_ori.mean(), advs_ori.std(), abs(advs_ori).max())

                mean_returns = np.mean(returns, axis = 0)
                print('    ', np.array_str(mean_returns, precision=3, suppress_small=True), np.sum(np.multiply(mean_returns, random_w)))
                
                lr_p, cr_p = 0.01, 0.3
                print(' ***** 2 update base policy using --w', random_w, ' --alpha: lr', lr_p, ' --clip range', cr_p, ' --minibatch', nbatch_train, ' --epoch', noptepochs)
                advs = compute_advs(advs_ori, random_w)
                mblossvals = nimibatch_update(  nbatch, noptepochs, nbatch_train,
                                                obs, returns, masks, actions, values, advs, neglogpacs,
                                                lr_p, cr_p, states, nsteps, model, update_type = 'all')   
                v_loss = mblossvals[1]  
                # v_loss = nimibatch_update(  nbatch, 15, nbatch_train/10,
                #                                 obs, returns, masks, actions, values, advs, neglogpacs,
                #                                 lr_p, cr_p, states, nsteps, model, update_type = 'value')  
                print(v_loss, abs(v_loss_pre - v_loss), mblossvals[2])
                display_updated_result( mblossvals, update, log_interval, nsteps, nbatch, 
                                    rewards, returns, advs, epinfos, model, logger)    

                v_loss_pre = v_loss

                # if g %2 == 1:
                #     print('save trajectiry')
                all_obs, all_a, all_ret = stack_values([all_obs, all_a, all_ret], [obs, actions, returns])
            save_model(model, 'r_'+str(i))

            print (' ***** 6 restore base policy params')
            model.replace_params(params_actor_base, params_type='actor')
            model.replace_params(params_value_base, params_type='value')  

        all_neglogpacs = model.get_neglogpac(all_obs, all_a)
        all_values = model.get_values(all_obs)

        all_advs_ori = all_ret - all_values
        all_advs = compute_advs(all_advs_ori, target_w)

        lr_s, cr_s = 0.0005, 0.2
        print('batch size', len(all_obs))
        print(' ***** 2 update base policy using --w', target_w, ' --alpha: lr', lr_s, ' --clip range', cr_s, ' --minibatch', nbatch_train*2, ' --epoch', noptepochs)
        mblossvals = nimibatch_update(  nbatch, noptepochs*2, nbatch_train*2,
                                        all_obs, all_ret, masks, all_a, all_values, all_advs, all_neglogpacs,
                                        lr_s, cr_s, states, nsteps, model, update_type = 'all') 
        print('value loss',mblossvals[1], 'entropy', mblossvals[2])
        # display_updated_result( mblossvals, update, log_interval, nsteps, nbatch, 
        #                     rewards_p, returns_p, advs_p, epinfos_p, model, logger)  



        obs, returns, masks, actions, values, rewards, neglogpacs, states, epinfos, ret = runner.run(int(nsteps*8), is_test=False) #pylint: disable=E0632
        print('collected', len(obs))
        advs_ori = returns - values
        advs = compute_advs(advs_ori, random_w)
        mblossvals = nimibatch_update(  nbatch, noptepochs, nbatch_train,
                                        obs, returns, masks, actions, values, advs, neglogpacs,
                                        lr_p, cr_p, states, nsteps, model, update_type = 'value')   

        mean_returns = np.mean(returns, axis = 0)
        print('after updated    ', np.array_str(mean_returns, precision=3, suppress_small=True), np.sum(np.multiply(mean_returns, target_w)))
        for r in range(REWARD_NUM):
            model.write_summary('meta/mean_ret_'+str(r+1), np.mean(returns[:,r]), update)
        model.write_summary('meta/mean_ret_sum', np.mean(np.sum(returns, axis=1), axis=0), update)


        print('\n') 

        if save_interval and (update % save_interval == 0 or update == 1 or update == nupdates) and logger.get_dir():
            save_model(model, update)
            save_model(model, 0)

        first = False
    env.close()

