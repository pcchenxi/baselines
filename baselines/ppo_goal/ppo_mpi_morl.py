import os
import time
import joblib
import numpy as np
import os.path as osp
import tensorflow as tf
from baselines import logger
from baselines.ppo_goal.common_functions import *
import matplotlib.pyplot as plt
from mpi4py import MPI
import datetime

import pygmo as pg


comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
nprocs = comm.Get_size()

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

    params_act_rand = model.get_current_params(params_type='actor')
    params_value_rand = model.get_current_params(params_type='value')
    params_pm_rand = model.get_current_params(params_type='forward')

    # checkdir = osp.join('../model', 'checkpoints')    
    # mocel_path = osp.join(checkdir, str('0'))    
    # model.load(mocel_path)
    params_pm = model.get_current_params(params_type='forward')
    params_value = model.get_current_params(params_type='value')

    model.replace_params(params_act_rand, params_type='actor') 
    model.replace_params(params_value_rand, params_type='value') 
    # # model.replace_params(params_pm_rand, params_type='forward') 
    if comm_rank == 0:
        save_model(model, 0)
        save_model(model, 1)
    # MPI.COMM_WORLD.gather(start_update) # wait to syn

    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)

    nupdates = total_timesteps//nbatch

    params_all_base = model.get_current_params(params_type='all')

    mode_name = ['av', 'pm', 'select']

    best_model = []
    best_score = 0
    mocel_path = []
    save_interval = 100
    save_count = 2*save_interval
    save_index = int(save_count/save_interval)

    select_count = 0

    policy_list_init = [0]
    policy_list = policy_list_init.copy()
    use_off_policy = False
    off_count = 0
    select_count = 0
    obs_buffer, obs_next_buffer, action_buffer, fixed_f_buffer, reward_buffer = [], [], [], [], []

    av_update_num = 1000
    av_count = 0
    pm_count = 0
    select_count = 0
    mode = 'pm'

    for update in range(start_update+1, nupdates+1):
        # model.replace_params(params_all_base, params_type='all')

        if mode == 'pm':
            pm_count += 1
        elif mode == 'av':
            av_count += 1
        elif mode == 'select':
            select_count += 1                    
        elif mode == 'trace':
            pm_count += 1   

        # collect trajectories
        mode = comm.bcast(mode, root=0)

        param_actor = []
        if mode == 'av':
            # mocel_path = '../model/checkpoints/'+str(0)  # use the current policy to sample trajectory for training policy
            # model.load(mocel_path)
            if False: #update%3 == 0:
                use_off_policy = True
                path_index = np.random.randint(0, save_index, 1)[0]
            else:
                use_off_policy = False
                path_index = 0

            mocel_path = '../model/checkpoints/'+str(path_index)
            model.load(mocel_path)
            model.replace_params(params_pm.copy(), params_type='forward') 
            model.replace_params(params_value.copy(), params_type='value')   

            obs, obs_next, returns, dones, actions, values, advs_ori, rewards, fixed_state_f, pm_states, neglogpacs, epinfos = runner.run(model, int(nsteps), use_off_policy=use_off_policy, random_prob = 1) #pylint: disable=E06327
        elif mode == 'pm' or mode == 'trace':
            path_index = pm_count-1
            mocel_path = '../model/checkpoints/'+str(path_index)
            model.load(mocel_path)
            model.replace_params(params_pm.copy(), params_type='forward') 
            model.replace_params(params_value.copy(), params_type='value')     

            obs, obs_next, returns, dones, actions, values, advs_ori, rewards, fixed_state_f, pm_states, neglogpacs, epinfos = runner.run(model, int(nsteps), use_off_policy=use_off_policy, random_prob = 1) #pylint: disable=E06327
        elif mode == 'select':
            path_index = select_count-1
            mocel_path = '../model/checkpoints/'+str(path_index)
            model.load(mocel_path)
            model.replace_params(params_pm.copy(), params_type='forward') 
            model.replace_params(params_value.copy(), params_type='value')     

            obs, obs_next, returns, dones, actions, values, advs_ori, rewards, fixed_state_f, pm_states, neglogpacs, epinfos = runner.run(model, int(nsteps*4), use_off_policy=use_off_policy, random_prob = 1) #pylint: disable=E06327

        print(mode, update, comm_rank, 'loading', mocel_path, save_index, rewards[:,-1].mean(), (rewards[:,-1]-rewards[:,0]-1).mean(), use_off_policy)
        # advs = advs_ori[:, -1] #+ advs_ori[:, 1]*2
        advs = np.asarray(advs_ori)

        obs_gather = MPI.COMM_WORLD.gather(obs)
        obs_next_gather = MPI.COMM_WORLD.gather(obs_next)
        pm_states_gather = MPI.COMM_WORLD.gather(pm_states)

        returns_gather = MPI.COMM_WORLD.gather(returns)
        actions_gather = MPI.COMM_WORLD.gather(actions)
        advs_gather = MPI.COMM_WORLD.gather(advs)
        values_gather = MPI.COMM_WORLD.gather(values)
        reward_gather = MPI.COMM_WORLD.gather(rewards)
        fixed_state_f_gather = MPI.COMM_WORLD.gather(fixed_state_f)
        neglogpacs_gather = MPI.COMM_WORLD.gather(neglogpacs)

        model_path_gather = MPI.COMM_WORLD.gather(mocel_path)
        score_gather = MPI.COMM_WORLD.gather(rewards[:,-1].mean())
        # param_actor_gather = MPI.COMM_WORLD.gather(param_actor)
        path_index_gather = MPI.COMM_WORLD.gather(path_index)


        if comm_rank == 0:
            length = len(obs)
            all_obs = np.asarray(obs_gather).reshape(nprocs*length, -1)
            all_obs_next = np.asarray(obs_next_gather).reshape(nprocs*length, -1)
            all_a = np.asarray(actions_gather).reshape(nprocs*length, -1)
            all_ret = np.asarray(returns_gather).reshape(nprocs*length, -1)
            all_adv = np.asarray(advs_gather).reshape(nprocs*length, -1)
            all_value = np.asarray(values_gather).reshape(nprocs*length, -1)
            all_reward = np.asarray(reward_gather).reshape(nprocs*length, -1)
            all_fsf = np.asarray(fixed_state_f_gather).reshape(nprocs*length, -1)
            # all_pm_state = np.asarray(pm_states_gather).reshape(nprocs*length, -1)
            all_neglogpacs = np.asarray(neglogpacs_gather).reshape(nprocs*length)


            print(all_obs.shape, all_a.shape, all_ret.shape, all_adv.shape, all_value.shape, all_reward.shape, all_fsf.shape, all_ret.shape, all_neglogpacs.shape)

            mean_returns = np.mean(all_ret, axis = 0)
            mean_values = np.mean(all_value, axis = 0)
            mean_rewards = np.mean(all_reward, axis = 0)
            mean_adv = np.mean(all_adv, axis = 0)

            print(update, mean_returns, mean_values, mean_rewards, mean_adv)

            lr_p, cr_p = 0.0003, 0.2

            print(mode, av_count, pm_count, select_count)
            if mode == 'av':
                mocel_path = '../model/checkpoints/'+str(0)  
                model.load(mocel_path)
                print('restore to current net for updating', mocel_path)                        
                print('in mode av, use_off_policy', use_off_policy) 
                mblossvals = nimibatch_update(  nbatch, noptepochs, nbatch_train,
                                            all_obs, all_obs_next, all_ret, all_a, all_value, all_adv[:,0], all_reward[:,0], all_fsf, all_neglogpacs,
                                            lr_p, cr_p, model, update_type = 'av', use_off_policy=use_off_policy)                                                
                display_updated_result( mblossvals, update, log_interval, nsteps, nbatch, 
                                all_reward, all_ret, all_value, all_adv, epinfos, model, logger) 
                # nimibatch_update_pm(noptepochs, nbatch_train, [all_obs, all_a, all_obs_next, all_fsf], all_reward[:,-1], lr_p, model, update, 'trace')
                
                save_model(model, save_index)
                save_count +=1

                save_model(model, 0)

                params_value = model.get_current_params(params_type='value')
                if av_count == av_update_num:
                    mode = 'pm'
                    pm_count = 0

            elif mode == 'pm' or mode == 'trace':
                if obs_buffer == []:
                    obs_buffer, obs_next_buffer, action_buffer, fixed_f_buffer, reward_buffer = all_obs.copy(), all_obs_next.copy(), all_a.copy(), all_fsf.copy(), all_reward.copy()
                    return_buffer = all_ret.copy()
                else:
                    obs_buffer = np.concatenate((obs_buffer, all_obs.copy()), axis=0)
                    obs_next_buffer = np.concatenate((obs_next_buffer, all_obs_next.copy()), axis=0)
                    action_buffer = np.concatenate((action_buffer, all_a.copy()), axis=0)
                    fixed_f_buffer = np.concatenate((fixed_f_buffer, all_fsf.copy()), axis=0)
                    reward_buffer = np.concatenate((reward_buffer, all_reward.copy()), axis=0)
                    return_buffer = np.concatenate((return_buffer, all_ret.copy()), axis=0)

                # if len(obs_buffer) > 300000:
                if pm_count == save_index:
                    print('in mode', mode, len(obs_buffer))
                    mocel_path = '../model/checkpoints/'+str(0)  
                    model.load(mocel_path)
                    print('restore to current net for updating', mocel_path)                

                    use_max = False

                    if mode == 'pm':
                        # reward_in = reward_buffer[:,-1] #(reward_buffer[:,0]+1)*10
                        iteration = 10
                    else:     
                        iteration = 30
                    reward_in = return_buffer[:,-1]
                    use_max = False
                    for i in range(iteration):
                        # if mode == 'trace' and i > 25:
                        #     use_max = True 
                    
                        nimibatch_update_pm(1, 128, [obs_buffer, action_buffer, obs_next_buffer, fixed_f_buffer], reward_in, lr_p, model, update, mode, use_max=use_max)
                        save_model(model, '0')
                        print(mode, i)
                        
                    save_model(model, save_index)
                    # # if pm_update_count < 5:
                    print('reset to random policy')
                    # model.replace_params(params_value_rand.copy(), params_type='value') 
                    model.replace_params(params_act_rand.copy(), params_type='actor') 

                    params_value = model.get_current_params(params_type='value')
                    params_pm = model.get_current_params(params_type='forward')

                    save_model(model, 0)

                    obs_buffer, obs_next_buffer, action_buffer, fixed_f_buffer = [], [], [], []

                    if mode == 'pm':
                        mode = 'trace'
                        pm_count = 0
                        # mode = 'select'
                        # select_count = 0                        
                    elif mode == 'trace':
                        # mode = 'select'
                        # select_count = 0
                        mode = 'av'
                        av_count = 0

                    best_score = 0
                    best_model = []

            elif mode == 'select':
                print('in mode select')
                score = np.asarray(score_gather).mean()
                # best_mean_index = np.argmax(score_gather)
                # print('score list', score_gather, score_gather[best_mean_index], best_score, best_mean_index)
                # policy_list.append(model_path_gather[best_mean_index])
                # print('policy list', policy_list)
                print('best model', best_model, best_score)
                if best_score < score:
                    # policy_list.append(best_mean_index)
                    best_score = score
                    best_model = model_path_gather[0]
                    model.load(best_model)  
                    # model.load_and_increase_logstd(best_model)  
                    model.replace_params(params_pm.copy(), params_type='forward') 
                    print('best model', best_model)
                    save_model(model, 0)
                
                if select_count == save_index:
                    mode = 'av'
                    av_count = 0

            print('\n') 
            save_index = int(save_count/save_interval)

            # print('send',params_pm[0])        

            # if save_interval and (update % save_interval == 0 or update == 1 or update == nupdates) and logger.get_dir():
            #     save_model(model, update)
            #     save_model(model, 0)
            #     print('save model')

            # first = False
            # params_all_base = model.get_current_params(params_type='all')

        # params_all_base = comm.bcast(params_all_base, root=0)
        mode = comm.bcast(mode, root=0)
        av_count = comm.bcast(av_count, root=0)
        pm_count = comm.bcast(pm_count, root=0)
        select_count = comm.bcast(select_count, root=0)

        save_index = comm.bcast(save_index, root=0)
        policy_list = comm.bcast(policy_list, root=0)

        params_pm = comm.bcast(params_pm.copy(), root=0)      
        params_value = comm.bcast(params_value.copy(), root=0)  

    env.close()


