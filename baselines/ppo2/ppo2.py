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

    start_update = 0

    # load training policy
    checkdir = osp.join('../model', 'checkpoints')    
    # checkdir = osp.join('../model', 'fun_models')
    mocel_path = osp.join(checkdir, str(start_update))

    if start_update != 0:
        model.load(mocel_path)
        pf_sets = np.load('../model/checkpoints/pf.npy').tolist()

    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)

    nupdates = total_timesteps//nbatch

    for update in range(start_update+1, nupdates+1):
        # assert nbatch % nminibatches == 0
        # nbatch_train = nbatch // nminibatches
        nbatch_train = nminibatches    
        tstart = time.time()

        # save the base policy params
        params_all_base = model.get_current_params(params_type='all')
        params_actor_base = model.get_current_params(params_type='actor')
        params_value_base = model.get_current_params(params_type='value')

        # prepare gradient buffer
        delta_actor_buff_1 = []
        delta_value_buff_1 = []

        delta_actor_buff = []
        delta_value_buff = []

        w_vs = []
        w_returns = []
        for i in range(3): 
            keep_weight = False
            print('--------------------------------', i, '--------------------------------')
            # 2: update base policy to a random direction
            # random_w = np.random.rand(REWARD_NUM)
            # random_w[1] = abs(random_w[1])
            # random_w[0] = abs(random_w[0])
            if i == 0:
                random_w = [1, 0,  0,  0, 0]
            elif i == 1:
                random_w = [0, 1,  0,  0, 0]
            elif i == 2:
                random_w = [0, 0,  1,  0, 0]
            # elif i == 3:
            #     random_w = [0, 0, -1,  0, 0]
            # elif i == 4:
            #     random_w = [0, 0,  0,  1, 0]
            # elif i == 5:
            #     random_w = [0, 0,  0, -1, 0]
 
 
            num_g = 15
            for g in range(num_g):

                print(' ***** 3 sample batch using updated policy with size', nsteps*nenvs, '    ', g)
                obs_p, returns_p, masks_p, actions_p, values_p, rewards_p, neglogpacs_p, states_p, epinfos_p, _ = runner.run(int(nsteps), is_test=False) #pylint: disable=E0632

                advs_p_ori = returns_p - values_p
                advs_p = compute_advs(advs_p_ori, random_w)
                print('    updated batch collected', values_p.shape)
                mean_returns = np.mean(returns_p, axis = 0)
                print('    ', np.array_str(mean_returns, precision=3, suppress_small=True), np.sum(np.multiply(mean_returns, random_w)))


                # update using new batch
                lr_s, cr_s = 0.0005, 0.2
                print(' ***** 4 update policy the second time', random_w, ' --beta: lr', lr_s, ' --clip range', cr_s, ' --minibatch', nbatch_train, ' --epoch', noptepochs)
                mblossvals = nimibatch_update(  nbatch, noptepochs, nbatch_train,
                                                obs_p, returns_p, masks_p, actions_p, values_p, advs_p, neglogpacs_p,
                                                lr_s, cr_s, states_p, nsteps, model, update_type = 'actor') 
                v_loss = nimibatch_update(  nbatch, 15, nbatch_train/10,
                                                obs_p, returns_p, masks_p, actions_p, values_p, advs_p, neglogpacs_p,
                                                lr_s, cr_s, states_p, nsteps, model, update_type = 'value') 
                print(v_loss[0])
                display_updated_result( mblossvals, update, log_interval, nsteps, nbatch, 
                                    rewards_p, returns_p, advs_p, epinfos_p, model, logger)   
                

            # params_actor_p = model.get_current_params(params_type='actor')
            # params_value_p = model.get_current_params(params_type='value')



            # obs_e, returns_e, masks_e, actions_e, values_e, rewards_e, neglogpacs_e, states_e, epinfos_e, _ = runner.run(int(100), is_test=False, use_deterministic=True) #pylint: disable=E0632
            mean_returns = np.mean(returns, axis = 0)
            w_v = np.sum(np.multiply(mean_returns, target_w))
            print(' eva', np.array_str(mean_returns, precision=3, suppress_small=True), w_v)
            w_vs.append(w_v)
            w_returns.append(mean_returns)

            # compute delta
            print(' ***** 5 compute and save dalta')
            keep_weight = update_pf_sets(pf_sets, np.mean(returns, axis = 0), v_loss)

            # if keep_weight:
            # delta_actor_params = model.get_current_params(params_type='actor') - params_actor_p
            # delta_value_params = model.get_current_params(params_type='value') - params_value_base
            # delta_value_params = params_value_p - params_value_base
            delta_actor_params = params_actor_p - params_actor_base
            delta_value_params = params_value_p - params_value_base        

            delta_actor_buff.append(delta_actor_params)
            delta_value_buff.append(delta_value_params)
            
            print (' ***** length of pf sets', len(pf_sets), keep_weight)


            # # for displaying one step updated returns
            for r_index in range(REWARD_NUM):
                model.write_summary('meta/mean_ret'+str(r_index+1), np.mean(returns[:,r_index]), update)
            # save the one step updated models
            # if keep_weight:
            save_model(model, 'r_'+str(i+1))
            np.save('../model/checkpoints/pf', np.asarray(pf_sets))


            #   6: restore base policy params
            print (' ***** 6 restore base policy params')
            model.replace_params(params_actor_base, params_type='actor')
            model.replace_params(params_value_base, params_type='value')     
                   
            # print(params_actor_base - model.get_current_params(params_type='actor'))



        # look ahead
                # 1: sample batch using base policy
        print(' ***** 1. collect batch using base policy with size', nsteps*nenvs)
        v_loss_pre = 0
        for k in range(5):
            obs, returns, masks, actions, values, rewards, neglogpacs, states, epinfos, ret = runner.run(int(nsteps), is_test=False) #pylint: disable=E0632
            advs_ori = returns - values
            print('    base batch collected', values.shape, k)

            mean_returns = np.mean(returns, axis = 0)
            print('    ', np.array_str(mean_returns, precision=3, suppress_small=True), np.sum(np.multiply(mean_returns, target_w)))
            
            lr_p, cr_p = 0.001, 0.2
            print(' ***** 2 update base policy using --w', target_w, ' --alpha: lr', lr_p, ' --clip range', cr_p, ' --minibatch', nbatch_train, ' --epoch', noptepochs)
            advs = compute_advs(advs_ori, target_w)
            mblossvals = nimibatch_update(  nbatch, noptepochs, nbatch_train,
                                            obs, returns, masks, actions, values, advs, neglogpacs,
                                            lr_p, cr_p, states, nsteps, model, update_type = 'actor')       
            v_loss = nimibatch_update(  nbatch, 15, nbatch_train/10,
                                            obs, returns, masks, actions, values, advs, neglogpacs,
                                            lr_p, cr_p, states, nsteps, model, update_type = 'value')  
            print(v_loss[0], abs(v_loss_pre - v_loss[0]), mblossvals[2])

            target_v = np.sum(np.multiply(mean_returns, target_w))
            if target_v > best_target_v:
                print('       >>>>>>>>>>>>  update best policy <<<<<<<<<<<<')
                best_target_v = target_v
                save_model(model, 'best')
                joblib.dump(params_all_base, '../model/checkpoints/best_base')

            # save_model(model, 't_'+str(k+1))

            display_updated_result( mblossvals, update, log_interval, nsteps, nbatch, 
                                rewards, returns, advs, epinfos, model, logger)   
            # if k!= 0 and abs(v_loss_pre - v_loss[0]) < 10:
            #     break
                
            v_loss_pre = v_loss[0]




        w_weights = compute_weights(w_vs*1, pf_sets, w_returns, target_w)
        print(' ***** 7 sum all delta params ----')
        for i in range(len(delta_actor_buff)):
            delta_actor_buff[i] = delta_actor_buff[i] * w_weights[i]
        sum_delta_actor = np.sum(delta_actor_buff, axis=0)
        sum_delta_value = np.sum(delta_value_buff, axis=0)

        print(' ***** 8 update policy -----')

        update_params_actor = params_actor_base + sum_delta_actor # + np.sum(delta_actor_buff_1, axis=0)
        update_params_value = params_value_base + sum_delta_value # + np.sum(delta_value_buff_1, axis=0)

        model.replace_params(update_params_actor, params_type='actor')
        model.replace_params(update_params_value, params_type='value')

        print(w_vs)
        print(w_weights)
        print('\n') 

        if save_interval and (update % save_interval == 0 or update == 1 or update == nupdates) and logger.get_dir():
            save_model(model, update)
            save_model(model, 0)

        first = False
    env.close()

