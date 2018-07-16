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


def learn(*, policy, env, nsteps, total_timesteps, ent_coef, lr,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
            save_interval=0):
    global min_return, max_return

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

    model = make_model('model', need_summary = True)
    model_pf = make_model('model_pf', need_summary = False)
    model_pf_2 = make_model('model_pf_2', need_summary = False)

    start_update = 0

    # load training policy
    checkdir = osp.join('../model', 'checkpoints')    
    checkdir_pf = osp.join('../model', 'fun_models')
    mocel_path = osp.join(checkdir, '%.5i'%(start_update))
    pf_path = checkdir_pf + '/hard_pushup_run2'
    pf_path_2 = checkdir_pf + '/easy_good'

    # model_pf.load(pf_path)
    # model_pf_2.load(pf_path_2)

    if start_update != 0:
        model.load(mocel_path)

    # model.load_value(checkdir+'/00088')

    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)
    epinfobuf = deque(maxlen=100)
    tfirststart = time.time()

    nupdates = total_timesteps//nbatch

    max_return = [200, 120, 0, 0, 0]

    for update in range(start_update+1, nupdates+1):
        assert nbatch % nminibatches == 0
        # nbatch_train = nbatch // nminibatches
        nbatch_train = nminibatches    
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        lrnow = frac*lr(frac)
        cliprangenow = cliprange(frac)

        lrnow = np.clip(lrnow, 1e-4, 1)
        cliprangenow = np.clip(cliprangenow, 0.1, 1)

        print(lrnow, cliprangenow)

        # lrnow = lr
        # cliprangenow = cliprange
        obs, returns, masks, actions, values, rewards, neglogpacs, states, epinfos, ret = runner.run(is_test=False) #pylint: disable=E0632
        pi_mean, pi_std = model.get_actions_dist(obs)

        advs_ori = returns - values
        # advs = np.sum(advs_ori, axis=1)
        # print(advs.shape)
        advs = compute_advs(max_return, values, advs_ori, [], [])

        # values_pf = model_pf.get_values(obs)
        # actions_suggest, _ = model_pf.get_actions_dist(obs)
        # adv_ori_suggest = values_pf - values
        # advs_suggest = compute_advs(max_return, values, adv_ori_suggest, np.abs(pi_mean - actions_suggest), pi_std)

        # values_pf_2 = model_pf_2.get_values(obs)
        # actions_suggest_2, _ = model_pf_2.get_actions_dist(obs)
        # adv_ori_suggest_2 = values_pf_2 - values
        # advs_suggest_2 = compute_advs(max_return, values, adv_ori_suggest_2, np.abs(pi_mean - actions_suggest_2), pi_std)    

        epinfobuf.extend(epinfos)
        mblossvals = []
        if states is None: # nonrecurrent version
            inds = np.arange(nbatch)
            for ep in range(noptepochs):
                print('ep', ep)
                np.random.shuffle(inds)
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, advs, neglogpacs))
                    losses = model.train(lrnow, cliprangenow, *slices)
                    # losses = model.train(lrnow, cliprangenow, *slices, actions_suggest_list = [actions_suggest[mbinds]], advs_suggest_list = [advs_suggest[mbinds]])
                    # losses = model.train(lrnow, cliprangenow, *slices, actions_suggest_list = [actions_suggest[mbinds], actions_suggest_2[mbinds]], advs_suggest_list = [advs_suggest[mbinds], advs_suggest_2[mbinds]])                    
                    mblossvals.append(losses)
                    # mblossvals.append(model.train(lrnow, cliprangenow, *slices))
        else: # recurrent version
            assert nenvs % nminibatches == 0
            envsperbatch = nenvs // nminibatches
            envinds = np.arange(nenvs)
            flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
            envsperbatch = nbatch_train // nsteps
            for ep in range(noptepochs):
                np.random.shuffle(envinds)
                for start in range(0, nenvs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mbflatinds = flatinds[mbenvinds].ravel()
                    slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mbstates = states[mbenvinds]
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))

        lossvals = np.mean(mblossvals, axis=0)
        tnow = time.time()
        fps = int(nbatch / (tnow - tstart))
        if update % log_interval == 0 or update == 1:
            print(returns.shape)
            # ev = explained_variance(values, returns)
            logger.logkv("serial_timesteps", update*nsteps)
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update*nbatch)
            logger.logkv("fps", fps)
            # logger.logkv("explained_variance", float(ev))
            # logger.logkv("return_mean", np.mean(returns))
            logger.logkv('rew_mean', np.mean(rewards))
            logger.logkv('eprewmean', safemean([epinfo['r'][-1] for epinfo in epinfobuf]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.logkv('time_elapsed', tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv(lossname, lossval)
            logger.dumpkvs()
            
            step_label = update*nbatch
            for i in range(REWARD_NUM):
                model.write_summary('return/mean_ret_'+str(i+1), np.mean(returns[:,i]), step_label)
                model.write_summary('sum_rewards/rewards_'+str(i+1), safemean([epinfo['r'][i] for epinfo in epinfobuf]), step_label)
                model.write_summary('mean_rewards/rewards_'+str(i+1), safemean([epinfo['r'][i]/epinfo['l'] for epinfo in epinfobuf]), step_label)
                # model.log_histogram('return_dist/return_'+str(i+1), returns[:,i], step_label)
                model.log_histogram('adv_dist/adv_'+str(i+1), advs_ori[:,i], step_label)
                
                # model.write_summary('adv_mean/adv_'+str(i+1), advs_ori[:,i].mean(), step_label)
                # model.write_summary('adv_std/adv_'+str(i+1), advs_ori[:,i].std(), step_label)
                
            model.write_summary('sum_rewards/score', safemean([epinfo['r'][-1] for epinfo in epinfobuf]), step_label)
            model.write_summary('sum_rewards/mean_length', safemean([epinfo['l'] for epinfo in epinfobuf]), step_label)

            for (lossval, lossname) in zip(lossvals, model.loss_names):
                model.write_summary('loss/'+lossname, lossval, step_label)

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

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)
