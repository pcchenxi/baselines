import os
import time
import joblib
import numpy as np
import os.path as osp
import tensorflow as tf
from baselines import logger
from baselines.ppo2.common_functions import *
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
    pf_sets = []
    pf_sets_convex = []
    target_w = np.zeros(REWARD_NUM)
    target_w[-1] = 1

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

    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)

    nupdates = total_timesteps//nbatch

    data = 0

    params_all_base = model.get_current_params(params_type='all')
    # params_actor_base = model.get_current_params(params_type='actor')
    params_value_base = model.get_current_params(params_type='value')

    pf_sets = []    
    max_hv = 0
    max_hv_set = []

    ref_point_min = np.full(REWARD_NUM, 10000)
    ref_point_max = np.full(REWARD_NUM, -10000)

    for update in range(start_update+1, nupdates+1):
        # if update < 2:
        max_hv = 0
        max_hv_set = []
        pf_sets = []

        model.replace_params(params_all_base, params_type='all') 
        params_value_base = model.get_current_params(params_type='value')

        t_b = time.time()
        # assert nbatch % nminibatches == 0
        # nbatch_train = nbatch // nminibatches
        nbatch_train = nminibatches    

        # np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)
        # random_w = np.random.rand(REWARD_NUM)
        random_w = np.zeros(REWARD_NUM)

        if comm_rank < REWARD_NUM:
            random_w[comm_rank%REWARD_NUM] = 1  
        else:
            random_w = np.random.rand(REWARD_NUM)

        # random_w = [0, 0, 1, 0, 0]

        num_g = 3 #np.random.randint(4) +1
        lr_p, cr_p = 0.0005, 0.2
        
        print(update, comm_rank, random_w, num_g)
        for g in range(num_g):
            # env.reset()
            t_m = time.time()
            obs, returns, masks, actions, values, advs_ori, rewards, neglogpacs, states, epinfos, ret = runner.run(int(nsteps), is_test=False) #pylint: disable=E0632
            # advs_ori = returns - values
            mean_returns = np.mean(returns, axis = 0)
            mean_values = np.mean(values, axis = 0)
            pf_sets.append(mean_returns)

            # mean_return_gather = MPI.COMM_WORLD.gather(mean_returns)
            # ws = MPI.COMM_WORLD.gather(random_w)
            if comm_rank == 0:             
                print(g, comm_rank, '    ', np.array_str(mean_returns, precision=3, suppress_small=True), np.sum(np.multiply(mean_returns, random_w)))
                print(update, comm_rank, '    ', np.array_str(mean_values, precision=3, suppress_small=True))

            advs = compute_advs(advs_ori, random_w)

            # lr_p, cr_p = 0.001, 0.5
            # print(g, comm_rank,' ***** 2 update base policy using --w', random_w, ' --alpha: lr', lr_p, ' --clip range', cr_p, ' --minibatch', nbatch_train, ' --epoch', noptepochs)
            mblossvals = nimibatch_update(  nbatch, noptepochs, nbatch_train,
                                            obs, returns, masks, actions, values, advs, neglogpacs,
                                            lr_p, cr_p, states, nsteps, model, update_type = 'all')   
            v_loss = mblossvals[1]  


        model.replace_params(params_value_base, params_type='value')  
        obs, returns, masks, actions, values, advs_ori, rewards, neglogpacs, states, epinfos, ret = runner.run(int(nsteps), is_test=False) #pylint: disable=E0632
        advs = compute_advs(advs_ori, target_w)

        obs_gather = MPI.COMM_WORLD.gather(obs)
        returns_gather = MPI.COMM_WORLD.gather(returns)
        actions_gather = MPI.COMM_WORLD.gather(actions)
        advs_gather = MPI.COMM_WORLD.gather(advs)
        values_gather = MPI.COMM_WORLD.gather(values)
        neglogpacs_gather = MPI.COMM_WORLD.gather(neglogpacs)

        ws = MPI.COMM_WORLD.gather(random_w)

        if comm_rank == 0:
            all_obs, all_a, all_ret, all_adv, all_value, all_neglogpacs = [], [], [], [], [], []
            for obs, actions, returns, advs, values, neglogpacs in zip(obs_gather, actions_gather, returns_gather, advs_gather, values_gather, neglogpacs_gather):
                for rw in range(REWARD_NUM):
                    model.write_summary('return/mean_ret_'+str(rw+1), np.mean(returns[:,rw]), update)
                all_obs, all_a, all_ret, all_adv, all_value, all_neglogpacs = stack_values([all_obs, all_a, all_ret, all_adv, all_value, all_neglogpacs], [obs, actions, returns, advs, values, neglogpacs])

            print (comm_rank, ' ***** 6 restore base policy params')
            model.replace_params(params_all_base, params_type='all')  

            ref_point_min = np.full(REWARD_NUM, 10000)
            ref_point_max = np.full(REWARD_NUM, -10000)

            min_ret = np.min(all_ret, axis=0)
            min_val = np.min(all_value, axis=0)
            ref_point_min = np.minimum(min_ret, ref_point_min)
            ref_point_min = np.minimum(min_val, ref_point_min)

            max_ret = np.max(all_ret, axis=0)
            max_val = np.max(all_value, axis=0)
            ref_point_max = np.maximum(max_ret, ref_point_max)
            ref_point_max = np.maximum(max_val, ref_point_max)

            all_adv = []
            for ret, value in zip(all_ret, all_value):
                ret_dif = (ret - ref_point_min)/(ref_point_max-ref_point_min)
                value_dif = (value - ref_point_min)/(ref_point_max-ref_point_min)

                hv_ret = 1
                for r in ret_dif:
                    hv_ret = hv_ret*r
                hv_value = 1
                for v in value_dif:
                    hv_value = hv_value*v

                adv =  (hv_ret - hv_value)
                all_adv.append(adv)
            all_adv = np.asarray(all_adv)

            all_neglogpacs = model.get_neglogpac(all_obs, all_a)
            all_values = model.value(all_obs)

            # all_adv_ori = all_ret - all_values
            # all_adv_t = compute_advs(all_adv_ori, target_w)
            # all_adv_t = (all_adv_t - all_adv_t.mean()) / (all_adv_t.std() + 1e-8)

            # all_adv += all_adv_t

            lr_s, cr_s = 0.0003, 0.2
            print('batch size', len(all_obs))
            print(' ***** 2 update base policy using --w', target_w, ' --alpha: lr', lr_s, ' --clip range', cr_s, ' --minibatch', nbatch_train, ' --epoch', noptepochs)
            mblossvals = nimibatch_update(  nbatch, noptepochs, nbatch_train*2,
                                            all_obs, all_ret, masks, all_a, all_values, all_adv, all_neglogpacs,
                                            lr_s, cr_s, states, nsteps, model, update_type = 'all', allow_early_stop = False) 
            print('value loss',mblossvals[1], 'entropy', mblossvals[2])
            

            obs, returns, masks, actions, values, advs_ori, rewards, neglogpacs, states, epinfos, ret = runner.run(int(nsteps), is_test=False) #pylint: disable=E0632

            display_updated_result( mblossvals, update, log_interval, nsteps, nbatch, 
                                rewards, returns, values, advs_ori, epinfos, model, logger)    

            params_all_base = model.get_current_params(params_type='all')


            mean_returns_t = np.mean(returns, axis = 0)
            print('after updated    ', np.array_str(mean_returns_t, precision=3, suppress_small=True), np.sum(np.multiply(mean_returns_t, random_w)))
            for r in range(REWARD_NUM):
                model.write_summary('meta/mean_ret_'+str(r+1), np.mean(returns[:,r]), update)


            print('\n') 

            if save_interval and (update % save_interval == 0 or update == 1 or update == nupdates) and logger.get_dir():
                save_model(model, update)
                save_model(model, 0)

        params_all_base = comm.bcast(params_all_base, root=0)
        first = False
    env.close()

