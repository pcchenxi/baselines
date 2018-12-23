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

    nenvs = 1
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

    model_name = '1'
    random_ws = np.load('/home/xi/workspace/weights_half.npy')
    w = random_ws[int(model_name)]
    print(int(model_name), w)
    
    model_path = '/home/xi/workspace/exp_models/exp_human/meta_finetune/' + str(model_name)
    model.load(model_path)

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
        obs, returns, masks, actions, values, rewards, neglogpacs, states, epinfos, ret = runner.run(nsteps, is_test=True, use_deterministic=True) #pylint: disable=E0632
        pi_mean, pi_std = model.get_actions_dist(obs)

        advs_ori = returns - values     

        print (epinfos)

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
        # print(rewards, values)
        print('mean rewards',np.mean(returns,axis=0))
        # print(np.max((rewards),axis=0))
        # print(np.min((rewards),axis=0))
        print('----------------------------------------------------------------------')
        print('length and score: ')
        scores = epinfos[-1]['r'][:-1]   
        scores_w = np.multiply(scores, w)
        print(len(values), epinfos[-1]['r'][-1])
        print('----------------------------------------------------------------------')
        print(w)
        print(epinfos[-1]['r'], np.sum(scores_w))
        if ret:
            break
    env.close()

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)