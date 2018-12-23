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
    plt.clf()

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

    model = make_model('model', need_summary = False)

    start_update = 0

    # load training policy
    checkdir = '/home/xi/workspace/exp_models/exp_ant/meta_finetune'   
    # checkdir = osp.join('../model', 'fun_models')
    mocel_path = osp.join(checkdir, str(start_update))

    if start_update != 0:
        model.load(mocel_path)
        # pf_sets = np.load('../model/checkpoints/pf.npy').tolist()
        # pf_sets_convex = np.load('../model/checkpoints/pf_convex.npy').tolist()

    checkdir = '/home/xi/workspace/model/exp_meta_morl/'   
    mocel_path = osp.join(checkdir, str(0))
    model.load(mocel_path) 

    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)
    
    nupdates = total_timesteps//nbatch
    sum_vs, indexs = [], []
    mp_mean, meta_mean = [], []
    mp_r, meta_r = [], []

    random_ws = np.load('/home/xi/workspace/weights_half.npy')
    # random_ws = np.load('/home/xi/workspace/ws.npy')

    obs, returns, masks, actions, values, rewards, neglogpacs, states, epinfos, ret = runner.run(int(nsteps), is_test=False, use_deterministic=True) #pylint: disable=E0632
    h1_all, h2_all, pi_all = [], [], []
    for update in range(0, 21, 4):
        random_w = np.ones(REWARD_NUM)
        random_w[1] = 0.05*update
        random_w[2] = 1 - random_w[1]
    #         
        print(random_w)
        plt.scatter(0, -400, c='white')
        plt.scatter(400, -0, c='white')        
        checkdir = '/home/xi/workspace/model/exp_meta_morl/'   
        mocel_path = osp.join(checkdir, str(update))
        model.load(mocel_path)  

        for ob in obs:
            h1, h2, pi = model.get_all_output([ob])
            h1_all.append(h1)
            h2_all.append(h2)
            pi_all.append(pi)
            # break
        
    mean_h1 = np.mean(pi_all, axis=0)
    std_h1 = np.std(pi_all, axis=0)
    print(np.array_str(std_h1, precision=3, suppress_small=True))

        # obs, returns, masks, actions, values, rewards, neglogpacs, states, epinfos, ret = runner.run(int(nsteps), is_test=False, use_deterministic=True) #pylint: disable=E0632
        # mean_returns = np.mean(returns, axis = 0)
        # mean_r = np.sum(np.multiply(mean_returns, random_w))
        # meta_r.append(mean_r)

        # meta_mean.append(mean_returns)
        # plt.scatter(np.asarray(meta_mean)[:,1], np.asarray(meta_mean)[:,2], c='b')

        # ---------------------------------------------------------------------------------------------
        # checkdir = '/home/xi/workspace/model/exp_mp_morl/'   
        # mocel_path = osp.join(checkdir, str(update))
        # model.load(mocel_path)  

        # obs, returns, masks, actions, values, rewards, neglogpacs, states, epinfos, ret = runner.run(int(nsteps), is_test=False, use_deterministic=True) #pylint: disable=E0632
        # mean_returns = np.mean(returns, axis = 0)
        # mean_r = np.sum(np.multiply(mean_returns, random_w))
        # mp_r.append(mean_r)

        # mp_mean.append(mean_returns)
        # plt.scatter(np.asarray(mp_mean)[:,1], np.asarray(mp_mean)[:,2], c='r', marker='x')

        # plt.show()

    # for i in range(len(mp_r)):
    #     print(meta_r[i], mp_r[i])

    # for update in range(30):
    #     checkdir = '/home/xi/workspace/exp_models/exp_ant/mp_train'   
    #     # checkdir = '/home/xi/workspace/model/checkpoints'     
    #     mocel_path = osp.join(checkdir, str(update))
    #     model.load(mocel_path)  

    #     if update == 30:
    #         random_w = np.ones(REWARD_NUM)
    #     else:
    #         random_w = random_ws[update][0:REWARD_NUM]
    #     # random_w = np.random.rand(REWARD_NUM)
    #     # random_w[0], random_w[3], random_w[4] = 1, 1, 1
    #     # random_w[1] = 0.05*update
    #     # random_w[2] = 1 - random_w[1]


    #     num_g = 1
    #     print(comm_rank, random_w)
    #     for g in range(num_g):
    #         obs, returns, masks, actions, values, rewards, neglogpacs, states, epinfos, ret = runner.run(int(nsteps), is_test=False, use_deterministic=True) #pylint: disable=E0632
    #         mean_returns = np.mean(returns, axis = 0)
    #         # if mean_returns[0] < 150:
    #         #     continue
    #         mean_return_gather = MPI.COMM_WORLD.gather(mean_returns)
    #         if comm_rank == 0:
    #             for mean_return in mean_return_gather:
    #                 # is_pf, is_convax, pf_sets, pf_sets_convex = update_pf_sets(pf_sets, mean_return, [])
    #                 # plt.scatter(np.asarray(mean_return)[1], np.asarray(mean_return)[2], c='r', marker='x')
    #                 plt.scatter(update, np.sum(np.multiply(mean_returns, random_w)), c='r', marker='x')

    #             print(g, comm_rank, '    ', np.array_str(mean_returns, precision=3, suppress_small=True), np.sum(np.multiply(mean_returns, random_w)))
    #             # if g%5 == 0:
    #             np.save('../model/checkpoints/pf_'+str(g)+'.npy', pf_sets)
    #     plt.pause(0.01)

    # plt.show()
    env.close()

