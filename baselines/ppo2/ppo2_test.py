import os
import time
import math
import joblib
import numpy as np
import os.path as osp
import tensorflow as tf
from baselines import logger
from collections import deque
from baselines.common import explained_variance
from baselines.ppo2.ratio_functions import *
from baselines.ppo2.common_functions import *


def learn(*, policy, env, nsteps, total_timesteps, ent_coef, lr,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
            save_interval=0):

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
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
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef, lr = lr(1),
                    max_grad_norm=max_grad_norm, need_summary = need_summary)

    if save_interval and logger.get_dir():
        import cloudpickle
        with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
            fh.write(cloudpickle.dumps(make_model))

    model = make_model('model', need_summary = False)
    # model_pf = make_model('model_pf', need_summary = False)
    # model_pf_2 = make_model('model_pf_2', need_summary = False)

    start_update = 1

    model_name = '0'
    
    # load training policy
    # checkdir = osp.join('../model', 'checkpoints')    
    checkdir = osp.join('../model/log/50', '0')
    model_path = osp.join(checkdir, model_name)
    # pf_path = checkdir_pf + '/easy_good'
    # pf_path_2 = checkdir_pf + '/14'

    model.load(model_path)
    # model_pf.load(pf_path)
    # model_pf_2.load(pf_path_2)

    # min_return, max_return = np.load(checkdir+ '/minmax_return_' + str(start_update) + '.npy')

    max_return = [200, 30, 0, 0, 0]
    # min_return = [0, 0, -80, -15, -0.2]
    # model.load_value(checkdir+'/1')

    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)

    epinfobuf = deque(maxlen=100)
    tfirststart = time.time()

    nupdates = total_timesteps//nbatch
    for update in range(start_update+1, nupdates+1):
        assert nbatch % nminibatches == 0
        # nbatch_train = nbatch // nminibatches
        nbatch_train = nminibatches    
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates

        lrnow = lr(frac)
        cliprangenow = cliprange(frac)

        # lrnow = lr
        # cliprangenow = cliprange
        obs, returns, masks, actions, values, rewards, neglogpacs, states, epinfos, ret = runner.run(nsteps, is_test=True, use_deterministic=False) #pylint: disable=E0632
        pi_mean, pi_std = model.get_actions_dist(obs)

        advs_ori = returns - values
        # advs = compute_advs(max_return, values, advs_ori, [], [])

        # values_pf = model_pf.get_values(obs)
        # actions_suggest, _ = model_pf.get_actions_dist(obs)
        # adv_ori_suggest = values_pf - values
        # advs_suggest = compute_advs(max_return, values, adv_ori_suggest, np.abs(pi_mean - actions_suggest), pi_std)


        # values_pf_2 = model_pf_2.get_values(obs)
        # actions_suggest_2 = model_pf_2.get_actions(obs)
        # adv_ori_suggest_2 = values_pf_2 - values
        # advs_suggest_2 = compute_advs(max_return, values, adv_ori_suggest_2)        

        print (epinfos)

        # for t in range(len(returns)):
            # print(epinfos[t])
        #     print(''
        #     , advs[t]
        #     , advs_suggest[t]
        #     # , advs_suggest_2[t]
        #     # , v_score_pf[t]
        #     # , np.array_str(np.multiply(advs_ori[t], dir_v[t])/dir_v_length[t], precision=3, suppress_small=True) \
        #     # , np.array_str(dir_v[t], precision=3, suppress_small=True) \
        #     , np.array_str(values[t], precision=3, suppress_small=True) \
        #     # # , np.array_str(ratios, precision=3, suppress_small=True) \
        #     # # , np.array_str(ratios_mean, precision=3, suppress_small=True) \
        #     , np.array_str(rewards[t], precision=3, suppress_small=True) \
        #     , np.array_str(returns[t], precision=3, suppress_small=True) \
        #     )

        # advs_normal = (advs - advs.mean()) / (advs.std() + 1e-8)
        epinfobuf.extend(epinfos)
        mblossvals = []
        # if states is None: # nonrecurrent version
        #     inds = np.arange(len(advs))
        #     for ep in range(noptepochs):
        #         print('ep', ep)
        #         np.random.shuffle(inds)
        #         for start in range(0, len(advs), nbatch_train):
        #             end = start + nbatch_train
        #             mbinds = inds[start:end]
        #             slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, advs, neglogpacs))
        #             # losses = model.train(lrnow, cliprangenow, *slices, [actions_suggest[mbinds], actions_suggest_2[mbinds]], [advs_suggest[mbinds], advs_suggest_2[mbinds]])
        #             losses = model.train(lrnow, cliprangenow, *slices, [actions_suggest[mbinds]], [advs_suggest[mbinds]])

        #             mblossvals.append(losses)

        # print(ratios)
     
        # print(advs_normal)
        # print(np.mean(rewards[:, 0:2],axis=0))
        print(np.max((rewards),axis=0))
        print(np.min((rewards),axis=0))
        print('----------------------------------------------------------------------')
        print('length and score: ')
        print(len(values), epinfos[-1]['r'][-1])
        print('----------------------------------------------------------------------')
        print(epinfos[-1]['r'])
        if ret:
            break
    env.close()

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)