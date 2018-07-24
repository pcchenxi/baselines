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

    if value_loss > 50:
        keep_weight = True 
    elif pf_sets == []:
        pf_stes.append(candidate_set)
        keep_weight = True 
    else:
        add_candidate = False
        for i in range(len(pf_sets)):
            if np.all(candidate_set > pf_sets): # pf_sets[i] is pareto dominated by the candidate, remove pf_sets[i] and add candidate to the pf_sets
                pf_sets.pop(i)
                add_candidate = True
            if np.any(candidate_set > pf_sets): # candidate is a part of current pf
                add_candidate = True
        if add_candidate:
            pf_sets.append()
            pf_stes.append(candidate_set)
            keep_weight = True 
        else:
            keep_weight = False

    return keep_weight


def learn(*, policy, env, nsteps, total_timesteps, ent_coef, lr,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
            save_interval=0):
    global min_return, max_return

    first = True
    pf_sets = []

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

        # save the base policy params
        params_actor_base = model.get_current_params(params_type='actor')
        params_value_base = model.get_current_params(params_type='value')

        # prepare gradient buffer
        delta_actor_buff = []
        delta_value_buff = []

        vote_1, vote_2 = 0, 0
        
        # # 1: sample batch using base policy
        # print(' ** 1. collect batch using base policy with size', nsteps*nenvs)
        # obs, returns, masks, actions, values, rewards, neglogpacs, states, epinfos, ret = runner.run(int(nsteps), is_test=False) #pylint: disable=E0632
        # advs_ori = returns - values
        # print('    base batch collected', values.shape)

        # advs = compute_advs(advs_ori, np.array([0, -1, 0, 0, 0]))

        # lr_p, cr_p = 5e-4, 0.2
        # mblossvals = nimibatch_update(  nbatch, noptepochs, nbatch_train,
        #                                 obs, returns, masks, actions, values, advs, neglogpacs,
        #                                 lr_p, cr_p, states, nsteps, model, update_type = 'all')       
        # display_updated_result( mblossvals, update, log_interval, nsteps, nbatch, 
        #                     rewards, returns, advs, epinfos, model, logger)   

        for i in range(2): 
            print('--------------------------------', i, '--------------------------------')
            # 2: update base policy to a random direction
            # random_w = np.random.rand(REWARD_NUM)
            if i%2 == 0:
                random_w = np.array([0, 1, 0, 0, 0])
                # random_w = np.array([0.1, 1, 0.1, 0.1, 0.1])
            elif i%2 != 0:
                random_w = np.array([0, -1, 0, 0, 0])
                # random_w = np.array([0.1, -1, 0.1, 0.1, 0.1])


            # 1: sample batch using base policy
            print(' ***** 1. collect batch using base policy with size', nsteps*nenvs)
            obs, returns, masks, actions, values, rewards, neglogpacs, states, epinfos, ret = runner.run(int(nsteps), is_test=False) #pylint: disable=E0632
            advs_ori = returns - values
            print('    base batch collected', values.shape)

            mean_returns = np.mean(returns, axis = 0)
            print('    ', np.array_str(mean_returns, precision=3, suppress_small=True))
            


            lr_p, cr_p = 0.001, 0.2
            print(' ***** 2 update base policy using --w', random_w, ' --alpha: lr', lr_p, ' --clip range', cr_p, ' --minibatch', nbatch_train, ' --epoch', noptepochs)
            advs = compute_advs(advs_ori, random_w)
            mblossvals = nimibatch_update(  nbatch, noptepochs, nbatch_train,
                                            obs, returns, masks, actions, values, advs, neglogpacs,
                                            lr_p, cr_p, states, nsteps, model, update_type = 'all')       
            params_actor_p = model.get_current_params(params_type='actor')
            params_value_p = model.get_current_params(params_type='value')
 



            for g in range(20):
                #   3: sample new batch
                print(' ***** 3 sample batch using updated policy with size', nsteps*nenvs, g)
                obs_p, returns_p, masks_p, actions_p, values_p, rewards_p, neglogpacs_p, states_p, epinfos_p, _ = runner.run(int(nsteps), is_test=False) #pylint: disable=E0632
                advs_p_ori = returns_p - values_p
                advs_p = compute_advs(advs_p_ori, random_w)
                print('    updated batch collected', values_p.shape)
                mean_returns = np.mean(returns_p, axis = 0)
                print('    ', np.array_str(mean_returns, precision=3, suppress_small=True))


                # update using new batch
                lr_s, cr_s = 0.0005, 0.2
                print(' ***** 4 update policy the second time', random_w, ' --beta: lr', lr_s, ' --clip range', cr_s, ' --minibatch', nbatch_train, ' --epoch', noptepochs)
                mblossvals = nimibatch_update(  nbatch, noptepochs, nbatch_train,
                                                obs_p, returns_p, masks_p, actions_p, values_p, advs_p, neglogpacs_p,
                                                lr_s, cr_s, states, nsteps, model, update_type = 'all') 

                display_updated_result( mblossvals, update, log_interval, nsteps, nbatch, 
                                    rewards_p, returns_p, advs_p, epinfos_p, model, logger)   

                obs_e, returns_e, masks_e, actions_e, values_e, rewards_e, neglogpacs_e, states_e, epinfos_e, _ = runner.run(int(100), is_test=False, use_deterministic=True) #pylint: disable=E0632
                print('    evaluating policy', )

            # # for displaying one step updated returns
            if i%2 == 0:
                for r_index in range(REWARD_NUM):
                    model.write_summary('meta_f/mean_ret'+str(r_index+1), np.mean(returns_e[:,r_index]), update)
            else:
                for r_index in range(REWARD_NUM):
                    model.write_summary('meta_b/mean_ret'+str(r_index+1), np.mean(returns_e[:,r_index]), update)
                    


            # compute delta
            print(' ***** 5 compute and save dalta')
            # keep_weight = update_pf_sets(pf_sets, np.mean(returns_p, axis = 0), mblossvals[1]):

            # if keep_weight:
            delta_actor_params = model.get_current_params(params_type='actor') - params_actor_p
            delta_value_params = model.get_current_params(params_type='value') - params_value_base
            # delta_value_params = params_value_p - params_value_base

            delta_actor_buff.append(delta_actor_params)
            delta_value_buff.append(delta_value_params)
            
            print (' ***** length of pf sets', len(pf_sets))

            # save the one step updated models
            if (update % save_interval == 0 or first) and i == 0:
                checkdir = osp.join('../model', 'checkpoints')
                savepath_base = osp.join(checkdir, '%.5i'%(1))
                print('    Saving to', savepath_base, savepath_base)
                model.save(savepath_base, savepath_base)
            if (update % save_interval == 0 or first) and i == 1:
                checkdir = osp.join('../model', 'checkpoints')
                savepath_base = osp.join(checkdir, '%.5i'%(2))
                print('    Saving to', savepath_base, savepath_base)
                model.save(savepath_base, savepath_base)




            #   6: restore base policy params
            print (' ***** 6 restore base policy params')
            model.replace_params(params_actor_base, params_type='actor')
            model.replace_params(params_value_base, params_type='value')     
                   
            # print(params_actor_base - model.get_current_params(params_type='actor'))




        print(' ***** 7 sum all delta params ----')
        sum_delta_actor = np.sum(delta_actor_buff, axis=0)
        sum_delta_value = np.sum(delta_value_buff, axis=0)

        print(' ***** 8 update policy -----')
        update_params_actor = params_actor_base + sum_delta_actor
        update_params_value = params_value_base + sum_delta_value

        model.replace_params(update_params_actor, params_type='actor')
        model.replace_params(update_params_value, params_type='value')

        print('\n') 

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

