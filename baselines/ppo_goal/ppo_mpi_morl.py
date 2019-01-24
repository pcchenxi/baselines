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
        save_model(model, 2)
    # MPI.COMM_WORLD.gather(start_update) # wait to syn

    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)

    nupdates = total_timesteps//nbatch

    mocel_path = []
    save_interval = 49
    save_count = 2*save_interval
    save_index = int(save_count/save_interval)

    obs_buffer, obs_next_buffer, action_buffer, fixed_f_buffer, reward_buffer = [], [], [], [], []

    av_update_num = 200
    av_count = 0
    pm_count = 0
    mode = 'pm'

    pm_buffer_obs = []
    pm_buffer_obs_next = []
    pm_buffer_a = []
    pm_buffer_fst = []
    pm_buffer_rew = []

    pm_cut, pm_scale, pm_mean = 0, 1, 0

    for update in range(start_update+1, nupdates+1):

        # if save_index%20 == 0:
        #     mode = 'pm'

        if mode == 'pm':
            pm_count += 1
        elif mode == 'av':
            av_count += 1               
        elif mode == 'trace':
            pm_count += 1   

        # collect trajectories
        mode = comm.bcast(mode, root=0)

        param_actor = []
        index_list = range(1, save_index+1, 1)
        if mode == 'av':
            #     path_index = np.random.choice(index_list, 1)[0]
            path_index = 0

        elif mode == 'pm' or mode == 'trace':
            path_index = (pm_count-1)%save_index
            path_index = index_list[path_index]

        mocel_path = '../model/checkpoints/'+ str(path_index)
        model.load(mocel_path)
        model.replace_params(params_pm.copy(), params_type='forward') 
        model.replace_params(params_value.copy(), params_type='value')     

        obs, obs_next, returns, dones, actions, values, advs_ori, rewards, fixed_state_f, pm_states, neglogpacs, epinfos = runner.run(model, int(nsteps), pm_cut, pm_scale)

        print(mode, update, comm_rank, 'loading', mocel_path, save_index, pm_cut, pm_scale)
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

            print(mode, av_count, pm_count)
            mocel_path = '../model/checkpoints/'+str(0)  
            model.load(mocel_path)                 
            print('restore to current net for updating', mocel_path)                        

            if mode == 'av':
                # all_neglogpacs = model.get_neglogpac(all_obs, all_a)
                mblossvals = nimibatch_update(  nbatch, noptepochs, nbatch_train,
                                            all_obs, all_obs_next, all_ret, all_a, all_value, all_adv[:,0], all_reward[:,0], all_fsf, all_neglogpacs,
                                            lr_p, cr_p, model, update_type = 'av')   

                display_updated_result( mblossvals, update, log_interval, nsteps, nbatch, 
                                all_reward, all_ret, all_value, all_adv, epinfos, model, logger) 
                
                save_count +=1
                if save_index != int(save_count/save_interval):
                    print('save !!!!!!!!!!!!!', len(pm_buffer_obs))
                    if pm_buffer_obs == []:
                        pm_buffer_obs, pm_buffer_obs_next, pm_buffer_a, pm_buffer_fst, pm_buffer_rew = all_obs.copy(), all_obs_next.copy(), all_a.copy(), all_fsf.copy(), all_reward.copy()
                    else:
                        pm_buffer_obs = np.concatenate((pm_buffer_obs, all_obs.copy()), axis=0)
                        pm_buffer_obs_next = np.concatenate((pm_buffer_obs_next, all_obs_next.copy()), axis=0)
                        pm_buffer_a = np.concatenate((pm_buffer_a, all_a.copy()), axis=0)
                        pm_buffer_fst = np.concatenate((pm_buffer_fst, all_fsf.copy()), axis=0)
                        pm_buffer_rew = np.concatenate((pm_buffer_rew, all_reward.copy()), axis=0)
              
                save_index = int(save_count/save_interval)

                save_model(model, save_index)
                save_model(model, 0)

                # if av_count % 150 == 0:
                #     mode = 'trace'
                #     pm_count = 0  
                if av_count == av_update_num:
                    mode = 'pm'
                    pm_count = 0

            elif mode == 'pm':
                print('in mode', mode, len(pm_buffer_obs))
                if pm_buffer_obs == []:
                    pm_buffer_obs, pm_buffer_obs_next, pm_buffer_a, pm_buffer_fst, pm_buffer_rew = all_obs.copy(), all_obs_next.copy(), all_a.copy(), all_fsf.copy(), all_reward.copy()
                else:
                    pm_buffer_obs = np.concatenate((pm_buffer_obs, all_obs.copy()), axis=0)
                    pm_buffer_obs_next = np.concatenate((pm_buffer_obs_next, all_obs_next.copy()), axis=0)
                    pm_buffer_a = np.concatenate((pm_buffer_a, all_a.copy()), axis=0)
                    pm_buffer_fst = np.concatenate((pm_buffer_fst, all_fsf.copy()), axis=0)
                    pm_buffer_rew = np.concatenate((pm_buffer_rew, all_reward.copy()), axis=0)

                if len(pm_buffer_obs) >= nprocs*length*save_index:
                    for i in range(30):
                        nimibatch_update_pm(2, 128, [pm_buffer_obs, pm_buffer_a, pm_buffer_obs_next, pm_buffer_fst], pm_buffer_rew[:,-1], lr_p, model, update, mode)
                        save_model(model, '0')
                        print(mode, i)

                    new_pm_error = model.pred_error(pm_buffer_obs, pm_buffer_a, pm_buffer_obs_next)
                    error_max = new_pm_error.max()
                    new_pm_error_act = runner.pm_activate(new_pm_error, error_max)

                    new_pm_error_with_rext = new_pm_error_act + pm_buffer_rew[:,0] +1

                    pm_mean = new_pm_error_with_rext.mean()

                    pm_cut = np.percentile(new_pm_error_with_rext, 70)
                    pm_scale = error_max
                    print('new pm error, mean_with_rext:', pm_mean, 'error max:', error_max, new_pm_error_with_rext.max(), 'cut:', pm_cut)
                    np.save('/home/xi/workspace/model/checkpoints/pm_norm', np.array(pm_cut, pm_scale))
                    mode = 'trace'
                    pm_count = 0                        
                    # mode = 'av'
                    # av_count = 0    

            elif mode == 'trace':
                if obs_buffer == []:
                    obs_buffer, obs_next_buffer, action_buffer, fixed_f_buffer, reward_buffer = all_obs.copy(), all_obs_next.copy(), all_a.copy(), all_fsf.copy(), all_reward.copy()
                    return_buffer = all_ret.copy()
                    values_buffer = all_value.copy()
                else:
                    obs_buffer = np.concatenate((obs_buffer, all_obs.copy()), axis=0)
                    obs_next_buffer = np.concatenate((obs_next_buffer, all_obs_next.copy()), axis=0)
                    action_buffer = np.concatenate((action_buffer, all_a.copy()), axis=0)
                    fixed_f_buffer = np.concatenate((fixed_f_buffer, all_fsf.copy()), axis=0)
                    reward_buffer = np.concatenate((reward_buffer, all_reward.copy()), axis=0)
                    return_buffer = np.concatenate((return_buffer, all_ret.copy()), axis=0)
                    values_buffer = np.concatenate((values_buffer, all_value.copy()), axis=0)

                if pm_count == save_index:         
                    reward_in = return_buffer[:,-1] # np.zeros(len(values_buffer))
                    for i in range(3):
                        nimibatch_update_pm(1, 128, [obs_buffer, action_buffer, obs_next_buffer, fixed_f_buffer], reward_in, lr_p, model, update, mode)
                        save_model(model, '0')
                        print(mode, i)
                        
                    save_model(model, save_index)
                    # # if pm_update_count < 5:
                    print('reset to random policy')
                    model.replace_params(params_value_rand.copy(), params_type='value') 
                    model.replace_params(params_act_rand.copy(), params_type='actor') 

                    obs_buffer, obs_next_buffer, action_buffer, fixed_f_buffer = [], [], [], []

                    mode = 'av'
                    av_count = 0

            params_value = model.get_current_params(params_type='value')
            params_pm = model.get_current_params(params_type='forward')

            save_model(model, 0)
            print('\n') 
            save_index = int(save_count/save_interval)
            print('pm_cut:', pm_cut, 'pm_max:', pm_scale, 'pm_mean', pm_mean)


        mode = comm.bcast(mode, root=0)
        av_count = comm.bcast(av_count, root=0)
        pm_count = comm.bcast(pm_count, root=0)

        pm_cut = comm.bcast(pm_cut, root=0)
        pm_scale = comm.bcast(pm_scale, root=0)

        save_index = comm.bcast(save_index, root=0)

        params_pm = comm.bcast(params_pm.copy(), root=0)      
        params_value = comm.bcast(params_value.copy(), root=0)  

    env.close()


