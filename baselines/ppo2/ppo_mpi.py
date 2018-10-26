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

def update_pf_sets(pf_sets, candidate_set, sub_target):
    keep_weight = False 
    pf_sets_new = []
    if pf_sets == []:
        pf_sets.append(candidate_set)
        keep_weight = True 
    else:
        add_candidate = True
        for i in range(len(pf_sets)):
            if np.all(np.asarray(candidate_set[1:3]) < np.asarray(pf_sets)[i][1:3]): # pf_sets[i] is pareto dominated by the candidate, remove pf_sets[i] and add candidate to the pf_set
                add_candidate = False
                break
            # elif np.any(np.asarray(candidate_set[1:3]) > np.asarray(pf_sets)[i][1:3]): # candidate is a part of current pf
            #     add_candidate = True
        if add_candidate:
            for i in range(len(pf_sets)):
                if np.all(np.asarray(candidate_set[1:3]) > np.asarray(pf_sets)[i][1:3]): # pf_sets[i] is pareto dominated by the candidate, remove pf_sets[i] and add candidate to the pf_set
                    pf_sets[i] = []

        if add_candidate:
            pf_sets.append(candidate_set)
            keep_weight = True 
        else:
            keep_weight = False    

        for i in range(len(pf_sets)):
            if pf_sets[i] != []:
                pf_sets_new.append(pf_sets[i])
        pf_sets = pf_sets_new*1

    # pf_sets[-1] = np.min(pf_sets, axis=0)
    is_convax = False
    pf_sets_convex = []
    if len(pf_sets)>6:
        set_cpy = pf_sets*1
        set_cpy.append(np.min(pf_sets, axis=0))
        index = ConvexHull(np.asarray(set_cpy)[:,1:3]).vertices        

        for i in index:
            if i < len(pf_sets):
                pf_sets_convex.append(pf_sets[i])
            if i == len(pf_sets)-1 and add_candidate:
                is_convax = True

        # print(index, len(pf_sets)-1, add_candidate)
        # pf_sets = pf_sets_new
        

        # print(len(pf_sets))
        # print(pf_sets)


    if len(pf_sets) > 1:
        # print(np.asarray(pf_sets)[:,1:3])
        plt.clf()
        plt.scatter(np.asarray(pf_sets)[:,1], np.asarray(pf_sets)[:,2], c='r')
        if pf_sets_convex != []:
            plt.scatter(np.asarray(pf_sets_convex)[:,1], np.asarray(pf_sets_convex)[:,2], c='pink')
        plt.scatter(candidate_set[1], candidate_set[2], c='b')
        if sub_target != []:
            plt.scatter(sub_target[1], sub_target[2], c='g')

        # pf_sets.append(np.min(pf_sets, axis=0))
        # hull = ConvexHull(np.asarray(pf_sets)[:,1:3])
        # index = ConvexHull(pf_sets).vertices
        # print(index)
        # plt.plot(np.asarray(pf_sets)[hull.vertices,1], np.asarray(pf_sets)[hull.vertices,2], 'b--', lw=2)
        # plt.plot(np.asarray(pf_sets)[hull.vertices[0],1], np.asarray(pf_sets)[hull.vertices[0],2], 'ro')

        plt.pause(0.01)
    return keep_weight, is_convax, pf_sets, pf_sets_convex


def compute_w_index(candidate_set, pf_sets, w):
    max_index = 0
    max_v = []
    for i in range(len(pf_sets)):
        pfset = pf_sets[i]
        v = np.sum(np.multiply(pfset, w))
        if max_v == [] or max_v < v:
            max_v = v 
            max_index = i
    # weight = pf_sets[max_index] - candidate_set

    if len(pf_sets) > 1:
        # print(np.asarray(pf_sets)[:,1:3])
        plt.clf()
        plt.scatter(np.asarray(pf_sets)[:,1], np.asarray(pf_sets)[:,2], c='r')
        plt.scatter(candidate_set[1], candidate_set[2], c='b')
        plt.scatter(np.asarray(pf_sets)[max_index,1], np.asarray(pf_sets)[max_index,2], c='g')
        plt.pause(0.01)
        # plt.show()   

    return pf_sets[max_index], max_v

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
    pf_sets_convex = []
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
        # pf_sets = np.load('../model/checkpoints/pf.npy').tolist()
        # pf_sets_convex = np.load('../model/checkpoints/pf_convex.npy').tolist()


    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)

    nupdates = total_timesteps//nbatch

    data = 0

    params_all_base = model.get_current_params(params_type='all')
    # params_actor_base = model.get_current_params(params_type='actor')
    # params_value_base = model.get_current_params(params_type='value')

    for update in range(start_update+1, nupdates+1):
        model.replace_params(params_all_base, params_type='all') 
        t_b = time.time()
        # assert nbatch % nminibatches == 0
        # nbatch_train = nbatch // nminibatches
        nbatch_train = nminibatches    

        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)
        task_random_w = np.random.rand(REWARD_NUM)
        task_random_w[0] = 1
        # index = np.random.choice(4, 1, p=[0.4, 0.3, 0.2, 0.1])
        index = np.random.choice(4, 1)
        if index == 0:
            random_w = [1, 0,  0,  0, 0]
            # random_w[1] = rand_num
        elif index == 1:
            random_w =[0, 1,  0,  0, 0]
        elif index == 2:
            random_w =[0, 0,  1,  0, 0]
        elif index == 3:
            random_w =[0, 0,  0,  1, 0]            
        else:
            random_w = np.random.rand(REWARD_NUM)
        # #     # random_w[0], random_w[3], random_w[4] = 1, 1, 1

        # if comm_rank == 0:
        #     random_w = [1, 1, 1, 1, 1]

        num_g = 10
        num_update_base = 5
        v_loss_pre = 0
        lr_p, cr_p = 0.001, 0.3
        print(comm_rank, random_w)
        for g in range(num_g):
            # env.reset()
            t_m = time.time()
            obs, returns, masks, actions, values, rewards, neglogpacs, states, epinfos, ret = runner.run(int(nsteps), is_test=False) #pylint: disable=E0632
            advs_ori = returns - values
            mean_returns = np.mean(returns, axis = 0)

            mean_return_gather = MPI.COMM_WORLD.gather(mean_returns)
            if comm_rank == 0 and update > 1:
                for mean_return in mean_return_gather:
                    is_pf, is_convax, pf_sets, pf_sets_convex = update_pf_sets(pf_sets, mean_return, [])
                print(g, task_random_w)
                print(comm_rank, '    ', np.array_str(mean_returns, precision=3, suppress_small=True), np.sum(np.multiply(mean_returns, random_w)))

            if g > num_update_base:
                advs = compute_advs(advs_ori, random_w)
            else:
                advs = compute_advs(advs_ori, task_random_w)

            if g < num_g:
                # lr_p, cr_p = 0.001, 0.5
                # print(g, comm_rank,' ***** 2 update base policy using --w', random_w, ' --alpha: lr', lr_p, ' --clip range', cr_p, ' --minibatch', nbatch_train, ' --epoch', noptepochs)
                mblossvals = nimibatch_update(  nbatch, noptepochs, nbatch_train,
                                                obs, returns, masks, actions, values, advs, neglogpacs,
                                                lr_p, cr_p, states, nsteps, model, update_type = 'all')   
                v_loss = mblossvals[1]  
                # v_loss = nimibatch_update(  nbatch, 10, 256,
                #                                 obs, returns, masks, actions, values, advs, neglogpacs,
                #                                 0.001, 0.2, states, nsteps, model, update_type = 'value')  
                # print(v_loss, abs(v_loss_pre - v_loss), mblossvals[2])
                # if comm_rank == 0:
                #     display_updated_result( mblossvals, update, log_interval, nsteps, nbatch, 
                #                         rewards, returns, advs, epinfos, model, logger)    

            # print(time.time() - t_m)
            v_loss_pre = v_loss
            lr_p, cr_p = lr_p*0.95, cr_p*0.95
            lr_p, cr_p = np.clip(lr_p, 0.0005, 10), np.clip(cr_p, 0.2, 10)
            # if mean_returns[2] > -30:
            #     break

            if g > num_update_base:
                obs_gather = MPI.COMM_WORLD.gather(obs)
                returns_gather = MPI.COMM_WORLD.gather(returns)
                actions_gather = MPI.COMM_WORLD.gather(actions)
                advs_gather = MPI.COMM_WORLD.gather(advs)

                if comm_rank == 0:
                    back_up_params = model.get_current_params(params_type='all')

                    # save_model(model, 1)
                    print(time.time() - t_b)

                    print(comm_rank, '     use w', random_w)
                    all_obs, all_a, all_ret, all_adv = [], [], [], []
                    for obs, actions, returns, advs in zip(obs_gather, actions_gather, returns_gather, advs_gather):
                        for rw in range(REWARD_NUM):
                            model.write_summary('return/mean_ret_'+str(rw+1), np.mean(returns[:,rw]), update)
                        all_obs, all_a, all_ret, all_adv = stack_values([all_obs, all_a, all_ret, all_adv], [obs, actions, returns, advs])
                    print('!!!!!!!!!!!!!!!!!!', len(all_obs))
                    # save_model(model, 'r_'+str(i))

                    print (comm_rank, ' ***** 6 restore base policy params')
                    # model.replace_params(params_actor_base, params_type='actor')
                    # model.replace_params(params_value_base, params_type='value')  
                    model.replace_params(params_all_base, params_type='all')  
                    # env.reset()

                    all_neglogpacs = model.get_neglogpac(all_obs, all_a)
                    all_values = model.get_values(all_obs)

                    # all_adv_ori = all_ret - all_values
                    # all_adv = compute_advs(all_adv_ori, target_w)
                    # all_adv = (all_adv - all_adv.mean()) / (all_adv.std() + 1e-8)

                    lr_s, cr_s = 0.0003, 0.2
                    print('batch size', len(all_obs))
                    print(' ***** 2 update base policy using --w', target_w, ' --alpha: lr', lr_s, ' --clip range', cr_s, ' --minibatch', nbatch_train, ' --epoch', noptepochs)
                    mblossvals = nimibatch_update(  nbatch, noptepochs, nbatch_train*2,
                                                    all_obs, all_ret, masks, all_a, all_values, all_adv, all_neglogpacs,
                                                    lr_s, cr_s, states, nsteps, model, update_type = 'all', allow_early_stop = False) 
                    print('value loss',mblossvals[1], 'entropy', mblossvals[2])
                    print(mblossvals)

                    obs, returns, masks, actions, values, rewards, neglogpacs, states, epinfos, ret = runner.run(int(nsteps), is_test=False) #pylint: disable=E0632
                    # print('collected', len(obs))
                    # advs_ori = returns - values
                    # advs = compute_advs(advs_ori, target_w)
                    # advs = (advs - advs.mean()) / (advs.std() + 1e-8)
                    # mblossvals = nimibatch_update(  nbatch, noptepochs, nbatch_train,
                    #                                 obs, returns, masks, actions, values, advs, neglogpacs,
                    #                                 lr_s, cr_s, states, nsteps, model, update_type = 'all')   
                    display_updated_result( mblossvals, update, log_interval, nsteps, nbatch, 
                                                        rewards, returns, advs, epinfos, model, logger)                                            
                    params_all_base = model.get_current_params(params_type='all')
                    model.replace_params(back_up_params, params_type='all')  


                    mean_returns_t = np.mean(returns, axis = 0)
                    # is_pf, is_convax, pf_sets, pf_sets_convex = update_pf_sets(pf_sets, mean_returns_t, [])
                    print('after updated    ', np.array_str(mean_returns_t, precision=3, suppress_small=True), np.sum(np.multiply(mean_returns_t, target_w)))
                    for r in range(REWARD_NUM):
                        model.write_summary('meta/mean_ret_'+str(r+1), np.mean(returns[:,r]), update)
                    if len(pf_sets) > 1:
                        _, target_v = compute_w_index(mean_returns_t, pf_sets, target_w)
                        # model.write_summary('meta/mean_ret_sum', np.mean(np.sum(returns, axis=1), axis=0), update)
                        model.write_summary('meta/mean_ret_sum', target_v, update)


                    print('\n') 

                    if save_interval and (update % save_interval == 0 or update == 1 or update == nupdates) and logger.get_dir():
                        save_model(model, update)
                        save_model(model, 0)
                        np.save('../model/checkpoints/pf.npy', pf_sets)
                        np.save('../model/checkpoints/pf_convex.npy', pf_sets_convex)

        params_all_base = comm.bcast(params_all_base, root=0)
        first = False
    env.close()
