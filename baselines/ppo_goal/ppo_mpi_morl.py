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
    model_pm = make_model('model_pm', need_summary = True)

    start_update = 0

    params_act_rand = model.get_current_params(params_type='actor')
    params_value_rand = model.get_current_params(params_type='value')
    params_pm_rand = model.get_current_params(params_type='forward')

    # if start_update != 0:
    # load training policy
    checkdir = osp.join('../model', 'checkpoints')    
    mocel_path = osp.join(checkdir, str('12_1'))    
    model.load(mocel_path)
    params_pm = model.get_current_params(params_type='forward')


    model.replace_params(params_act_rand, params_type='actor') 
    model.replace_params(params_value_rand, params_type='value') 
    # model.replace_params(params_pm_rand, params_type='forward') 
    save_model(model, 0)

    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)

    nupdates = total_timesteps//nbatch

    params_all_base = model.get_current_params(params_type='all')
    num_saved = int(start_update/save_interval) + 1

    model_list = ['0_0', '1_0', '2_0', '3_0', '4_0', '5_0', '6_0', '7_0', '8_0', '9_0', '10_0', '11_0', '12_0']

    max_rew = 0
    if comm_rank == 0:
        best_model, max_rew = [], 0
        for load_index in model_list:
            mocel_path = '/home/xi/workspace/model/checkpoints/'+str(load_index)
            model.load(mocel_path)
            # model.load_and_increase_logstd(mocel_path)
            model.replace_params(params_pm, params_type='forward') 
            obs, obs_next, returns, dones, actions, values, advs_ori, rewards, neglogpacs, epinfos = runner.run(model, int(1024), is_test=False, random_prob = 1) #pylint: disable=E0632

            mean_rew = rewards[:,1].mean()
            if mean_rew > max_rew:
                best_model = mocel_path
                max_rew = mean_rew
                print(load_index, mean_rew)
        
        model.load_and_increase_logstd(best_model)
        model.replace_params(params_pm, params_type='forward') 
        save_model(model, 0)

    MPI.COMM_WORLD.gather(max_rew)
    #     params_all_base = model.get_current_params(params_type='all')
    # params_all_base = comm.bcast(params_all_base, root=0)

    for update in range(start_update+1, nupdates+1):
        model.replace_params(params_all_base, params_type='all') 

        load_index = np.random.choice(model_list, 1)[0]
        mocel_path = '../model/checkpoints/'+str(load_index)
        model.load(mocel_path)
        model.replace_params(params_pm, params_type='forward') 

        runner.run(model, int(nsteps), is_test=False) #pylint: disable=E0632
        
        mocel_path = '../model/checkpoints/'+str(0)
        model.load(mocel_path)

        obs, obs_next, returns, dones, actions, values, advs_ori, rewards, neglogpacs, epinfos = runner.run(model, int(nsteps), is_test=False, random_prob = 0.7) #pylint: disable=E06327

        # load_index = np.random.choice(model_list, 1)[0]
        # mocel_path = '../model/checkpoints/'+str(load_index)
        # model.load(mocel_path)
        # obs, obs_next, returns, dones, actions, values, advs_ori, rewards, neglogpacs, epinfos = runner.run(model, int(nsteps), is_test=False, random_prob = 1) #pylint: disable=E0632

        # mocel_path = '../model/checkpoints/'+str('0')
        # model.load(mocel_path)


        # advs = advs_ori[:, -1] #+ advs_ori[:, 1]*2
        advs = np.asarray(advs_ori)

        obs_gather = MPI.COMM_WORLD.gather(obs)
        obs_next_gather = MPI.COMM_WORLD.gather(obs_next)

        returns_gather = MPI.COMM_WORLD.gather(returns)
        actions_gather = MPI.COMM_WORLD.gather(actions)
        advs_gather = MPI.COMM_WORLD.gather(advs)
        values_gather = MPI.COMM_WORLD.gather(values)
        reward_gather = MPI.COMM_WORLD.gather(rewards)
        neglogpacs_gather = MPI.COMM_WORLD.gather(neglogpacs)

        if comm_rank == 0:
            all_obs, all_obs_next, all_a, all_ret, all_adv, all_value, all_reward, all_neglogpacs = [], [], [], [], [], [], [], []
            for obs, obs_next, actions, returns, advs, values, rewards, neglogpacs in zip(obs_gather, obs_next_gather, actions_gather, returns_gather, advs_gather, values_gather, reward_gather, neglogpacs_gather):
                all_obs, all_obs_next, all_a, all_ret, all_adv, all_value, all_reward, all_neglogpacs = stack_values([all_obs, all_obs_next, all_a, all_ret, all_adv, all_value, all_reward, all_neglogpacs], [obs, obs_next, actions, returns, advs, values, rewards, neglogpacs])

            print(all_obs.shape, all_ret.shape)

            mean_returns = np.mean(all_ret, axis = 0)
            mean_values = np.mean(all_value, axis = 0)
            mean_rewards = np.mean(all_reward, axis = 0)
            mean_adv = np.mean(all_adv, axis = 0)

            runner.min_value = mean_rewards[-1]
            print(mean_returns, mean_values, mean_rewards, mean_adv)

            # print(obs.shape, advs.shape, returns.shape)
            # print(update, comm_rank, 'ret    ', np.array_str(mean_returns, precision=3, suppress_small=True))
            # print(update, comm_rank, 'val    ', np.array_str(mean_values, precision=3, suppress_small=True))
            # print(update, comm_rank, 'adv    ', np.array_str(mean_rewards, precision=3, suppress_small=True))
            # print(values_ret[:,-1].mean())

            lr_p, cr_p = 0.0003, 0.2

            mblossvals = nimibatch_update(  nbatch, noptepochs, nbatch_train,
                                            all_obs, all_obs_next, all_ret, all_a, all_value, all_adv[:,-1], all_neglogpacs,
                                            lr_p, cr_p, nsteps, model, update_type = 'av')   
            print(mblossvals[0], mblossvals[1], mblossvals[2])
            display_updated_result( mblossvals, update, log_interval, nsteps, nbatch, 
                                all_reward, all_ret, all_value, all_adv, epinfos, model, logger)    


            print('\n') 

            if save_interval and (update % save_interval == 0 or update == 1 or update == nupdates) and logger.get_dir():
                save_model(model, update)
                save_model(model, 0)
                num_saved += 1

            first = False
            params_all_base = model.get_current_params(params_type='all')

        params_all_base = comm.bcast(params_all_base, root=0)
    env.close()


