import os
import time
import joblib
import math
import numpy as np
import os.path as osp
import tensorflow as tf
from baselines import logger
from collections import deque
from baselines.ppo2.ratio_functions import *
from baselines.ppo2.common_functions import *
import matplotlib.pyplot as plt
from mpi4py import MPI
import datetime
import scipy
import pygmo as pg


def learn(*, policy, env, nsteps, total_timesteps, ent_coef, lr,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
            save_interval=0):

    first = True

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

    weight = np.ones(REWARD_NUM)
    weight = [0, 0, 0, 1]
    ref_point_min = np.full(REWARD_NUM, 10000)
    ref_point_max = np.full(REWARD_NUM, -10000)

    for update in range(start_update+1, nupdates+1):
        obs, obs_next, returns, dones, actions, values, advs_ori, rewards, neglogpacs, states, epinfos, ret = runner.run(int(nsteps), is_test=False) #pylint: disable=E0632
        advs = advs_ori[:, -1] #+ advs_ori[:, -2]*0.3
        # advs = compute_advs(advs_ori, [1, -1, 1, 1, 1, 0])

        # for i in range(len(obs)):
        #     print(returns[i], rewards[i], values[i], masks[i])
        # obs_ret = np.concatenate((obs, actions), axis=1)
        # values_ret = model.value_ret(obs_ret)

        # advs_ori = returns - values

        # # q_loss = values_ret - returns
        # # q_loss_mean = np.mean(q_loss, axis = 0)
        # # print('Q loss', q_loss_mean)
        # # print('V loss', np.mean(advs_ori, axis = 0))

        # for i in range(len(values)):
        #     print(advs_ori[i], values_ret[i] - values[i], values_ret[i], returns[i])


        # # # advs = np.max(advs_ori, axis=1)
        # # # print(advs.shape)

        # if update < 30:
        #     ref_point_min = np.full(REWARD_NUM, 10000)
        #     ref_point_max = np.full(REWARD_NUM, -10000)

        # advs, advs_scale = [], []
        # min_ret = np.min(returns, axis=0)
        # min_val = np.min(values, axis=0)
        # ref_point_min = np.minimum(min_ret, ref_point_min)
        # ref_point_min = np.minimum(min_val, ref_point_min)

        # max_ret = np.max(returns, axis=0)
        # max_val = np.max(values, axis=0)
        # # ref_point_max = np.maximum(max_ret, ref_point_max)
        # ref_point_max = np.maximum(max_val, ref_point_max)

        # print(ref_point_min)
        # print(ref_point_max)

        # for ret, value in zip(returns, values):
        #     ret_dif = (ret - ref_point_min)/(ref_point_max-ref_point_min)
        #     value_dif = (value - ref_point_min)/(ref_point_max-ref_point_min)

        #     adv_scale = ret_dif - value_dif
        #     advs_scale.append(adv_scale)

        #     # compare = value_dif - value_dif[-1]
        #     # max_index = np.argmin(compare)

        #     hv_ret = 1
        #     for r in ret_dif:
        #         hv_ret = hv_ret*r
        #     hv_value = 1
        #     for v in value_dif:
        #         hv_value = hv_value*v

        #     adv =  (hv_ret - hv_value)
        #     advs.append(adv)

        # advs_scale = np.asarray(advs_scale)
        # advs_scale = np.clip(advs_scale, 0, 100)
        # print(np.sum(advs_scale, axis = 0))

        # for ret, value in zip(returns, values):
        #     ret_dif = (ret - ref_point_min)/(ref_point_max-ref_point_min)
        #     value_dif = (value - ref_point_min)/(ref_point_max-ref_point_min)
        #     hv = pg.hypervolume(-np.asarray([ret_dif, value_dif]))
        #     adv = hv.exclusive(0, np.ones(REWARD_NUM)) #+ (ret - value)[-1]
        #     if adv == 0:
        #         hv_ret = 1
        #         for r in ret_dif:
        #             hv_ret = hv_ret*r
        #         hv_value = 1
        #         for v in value_dif:
        #             hv_value = hv_value*v
        #         adv =  (hv_ret - hv_value)

        #     advs.append(adv)
        #     # print(adv, hv.compute([100, 1000, 100, 1000]), hv.exclusive(1, [100, 1000, 100, 1000]))
        #     # hv_index = hv.compute(np.array([100, 1000, 100, 1000]))


        # advs_scaled = []
        # for ret, value in zip(returns, values):
        #     ret_scaled = (ret - ref_point_min)/(value-ref_point_min)
        #     adv_scaled = ret_scaled - np.ones(REWARD_NUM)
        #     advs_scaled.append(adv_scaled)

        # min_adv = np.min(advs_scaled, axis=0)
        # for adv, value in zip(advs_scaled, values):
        #     print(adv)
        #     adv_diff = adv - min_adv
        #     hv_adv = 1
        #     for r in adv_diff:
        #         hv_adv = hv_adv*r
        #     advs.append(hv_adv)

        advs = np.asarray(advs)
        print('advs mean', advs.mean(), advs.max(), advs.min())

        # obs_ret = returns[:, 0:-1] #np.concatenate((obs, returns[:, 0:-1]), axis=1)
        # values_ret = model.value_ret(obs_ret)
        # advs = values_ret[:,-1] - values[:,-1] #+ advs_ori[:, -1]
        # advs = advs_ret[:, -1]
        # advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        # for ret, adv_ori in zip(returns[:,-1], advs_ori):
        #     print(adv_ori)

        # for i in range(len(advs)):
            # print(values_ret[i,-1], values[i,-1], returns[i])
        # cm_adv = np.corrcoef(values_ret[:,-1], returns[:,-1])
        # print(cm_adv)
        # print('shape', advs.shape, advs.mean(), advs.std())

        # quality_weight = []
        # rewards_around = np.around(rewards, decimals=3)
        # for i in range(REWARD_NUM):
        #     d = rewards_around[:, i]
        #     hist, bin_edges = np.histogram(d, bins=1000)
        #     entr = scipy.stats.entropy(hist/np.sum(hist))
        #     # print(i, entr)
        #     # print(hist)
        #     quality = sum(hist > 0)
        #     quality_weight.append(quality)

        # quality_weight = np.asarray(quality_weight)
        # quality_weight = quality_weight/quality_weight.max()
        # print(quality_weight)

        # # # adv_clip = np.clip(advs_ori, 0, 1000000)
        # # adv_goodness = np.std(rewards, axis=0)
        # # print(adv_goodness)
        # # # best_cr = -10
        # # # for _ in range(500):
        # # #     weight = np.random.rand(REWARD_NUM) *adv_goodness
        # # #     weight[-1] = 0
        # # #     advs = compute_advs(advs_ori, weight)
        # # #     advs_g = np.clip(advs_ori[:, -1].transpose(), -0.0001, 1000)
        # # #     cm = np.corrcoef([advs.transpose(), advs_g])[1, 0]
        # # #     if cm > best_cr:
        # # #         best_cr = cm
        # # #         best_weight = weight
        # # advs_g = advs_ori.copy()
        # advs_g = (advs_ori - np.mean(advs_ori, axis=0)) / (np.std(advs_ori, axis=0) + 1e-8)
        # # advs_ = np.clip(advs_ori, -0.0001, 1000)
        # # advs_g = np.concatenate((advs_g, np.asarray([returns[:, -1]]).T), axis=1)
        # # print(advs_g.shape)
        # cm_adv = np.corrcoef(advs_g.transpose())
        # cm_ret = np.corrcoef(returns.transpose())

        # weight_ret = np.clip(cm_ret[-1], 0, 1)

        # # weight = np.zeros(REWARD_NUM)
        # # for i in range(REWARD_NUM):
        # #     weight += weight_ret[i] * np.clip(cm_adv[i], 0, 1) 
        # # weight *= quality_weight
        # print(cm_ret)
        # print(cm_adv)
        # # print(best_cr, best_weight)
        # # weight = cm_ret[-1] * quality_weight
        # # weight = np.clip(weight, 0, 1)
        # # weight = weight/weight.max()
        # print('use weight', weight)

        # for rw in range(REWARD_NUM):
        #     model.write_summary('weight/_'+str(rw+1), weight[rw], update)

        # advs = compute_advs(advs_ori, weight)

        mean_returns = np.mean(returns, axis = 0)
        mean_values = np.mean(values, axis = 0)
        mean_rewards = np.mean(advs_ori, axis = 0)

        print(obs.shape, advs.shape)
        print(update, comm_rank, 'ret    ', np.array_str(mean_returns, precision=3, suppress_small=True))
        print(update, comm_rank, 'val    ', np.array_str(mean_values, precision=3, suppress_small=True))
        print(update, comm_rank, 'adv    ', np.array_str(mean_rewards, precision=3, suppress_small=True))
        # print(values_ret[:,-1].mean())

        advs_g = np.concatenate((advs_ori, np.asarray([advs]).T), axis=1)
        cm = np.corrcoef(advs_g.transpose())
        print(cm)

        lr_p, cr_p = 0.0003, 0.2
        # max_val = np.max(values_next, axis=0)
        # min_val = np.min(values_next, axis=0)
        # runner.ref_point_max = np.maximum(max_val[:-1], runner.ref_point_max)
        # runner.ref_point_min = np.minimum(min_val[:-1], runner.ref_point_min)

        mblossvals = nimibatch_update(  nbatch, noptepochs, nbatch_train,
                                        obs, obs_next, returns, actions, values, advs, rewards, neglogpacs,
                                        lr_p, cr_p, states, nsteps, model, update_type = 'all')   
        display_updated_result( mblossvals, update, log_interval, nsteps, nbatch, 
                            rewards, returns, values, advs_ori, epinfos, model, logger)    


        print('\n') 

        if save_interval and (update % save_interval == 0 or update == 1 or update == nupdates) and logger.get_dir():
            save_model(model, update)
            save_model(model, 0)

        first = False
    env.close()

