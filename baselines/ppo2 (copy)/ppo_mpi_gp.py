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
import sklearn.gaussian_process as gp

from scipy.stats import norm
from scipy.optimize import minimize

import collections



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
def expected_improvement(x, gaussian_process, evaluated_loss, greater_is_better=True, n_params=1):
    """ expected_improvement
    Expected improvement acquisition function.
    Arguments:
    ----------
        x: array-like, shape = [n_samples, n_hyperparams]
            The point for which the expected improvement needs to be computed.
        gaussian_process: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.
        evaluated_loss: Numpy array.
            Numpy array that contains the values off the loss function for the previously
            evaluated hyperparameters.
        greater_is_better: Boolean.
            Boolean flag that indicates whether the loss function is to be maximised or minimised.
        n_params: int.
            Dimension of the hyperparameter space.
    """

    x_to_predict = x.reshape(-1, n_params)

    mu, sigma = gaussian_process.predict(x_to_predict, return_std=True)

    if greater_is_better:
        loss_optimum = np.max(evaluated_loss)
    else:
        loss_optimum = np.min(evaluated_loss)

    scaling_factor = (-1) ** (not greater_is_better)

    # In case sigma equals zero
    with np.errstate(divide='ignore'):
        Z = scaling_factor * (mu - loss_optimum) / sigma
        expected_improvement = scaling_factor * (mu - loss_optimum) * norm.cdf(Z) + sigma * norm.pdf(Z)
        expected_improvement[sigma == 0.0] == 0.0

    return -1 * expected_improvement


def sample_next_hyperparameter(acquisition_func, gaussian_process, evaluated_loss, greater_is_better=False,
                               bounds=(0, 10), n_restarts=25):
    """ sample_next_hyperparameter
    Proposes the next hyperparameter to sample the loss function for.
    Arguments:
    ----------
        acquisition_func: function.
            Acquisition function to optimise.
        gaussian_process: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.
        evaluated_loss: array-like, shape = [n_obs,]
            Numpy array that contains the values off the loss function for the previously
            evaluated hyperparameters.
        greater_is_better: Boolean.
            Boolean flag that indicates whether the loss function is to be maximised or minimised.
        bounds: Tuple.
            Bounds for the L-BFGS optimiser.
        n_restarts: integer.
            Number of times to run the minimiser with different starting points.
    """
    best_x = None
    best_acquisition_value = 1
    n_params = bounds.shape[0]

    for starting_point in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, n_params)):

        res = minimize(fun=acquisition_func,
                       x0=starting_point.reshape(1, -1),
                       bounds=bounds,
                       method='L-BFGS-B',
                       args=(gaussian_process, evaluated_loss, greater_is_better, n_params))

        if res.fun < best_acquisition_value:
            best_acquisition_value = res.fun
            best_x = res.x

    return best_x

def learn(*, policy, env, nsteps, total_timesteps, ent_coef, lr,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
            save_interval=0):

    first = True
    pf_sets = []
    pf_sets_convex = []
    target_w = [0.0, 0.0, 0.0, 0.0, 1.0]
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

    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)

    nupdates = total_timesteps//nbatch

    data = 0

    params_all_base = model.get_current_params(params_type='all')
    # params_actor_base = model.get_current_params(params_type='actor')
    # params_value_base = model.get_current_params(params_type='value')

    bounds = np.array([[0, 0.01], [0, 1], [0, 1], [0, 0.01], [0, 1]])

    for update in range(start_update+1, nupdates+1):
        model.replace_params(params_all_base, params_type='all') 
        t_b = time.time()
        # assert nbatch % nminibatches == 0
        # nbatch_train = nbatch // nminibatches
        nbatch_train = nminibatches    

        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)
        random_w = np.random.rand(REWARD_NUM)
        # random_w = np.random.normal(np.zeros(REWARD_NUM), np.full(REWARD_NUM, 0.1))
        if comm_rank == 0:
            random_w = target_w            

        # random_w[0] = 0
        # random_w = np.clip(random_w, 0, 1)

        # base = 2
        # if comm_rank%base == 0:
        #     random_w = [0, 0, 1, 0, 0]
        # elif comm_rank%base == 1:
        #     random_w = [0, 1, 0, 0, 0]
        # elif comm_rank%base == 1:
        #     random_w = [0, 0, 1, 0, 0]     
        # random_w = [0, 0, 1, 0]

        # random_w = [1, 1, 1, 1]
        # random_w[0], random_w[3], random_w[4] = 1, 1, 1
        # random_w[1] = 1 #0.05*comm_rank
        # random_w[2] = 1 - random_w[1]


        num_g = 1 #np.random.randint(4) +1
        v_loss_pre = 0
        lr_p, cr_p = 0.0003, 0.2
        lr_s, cr_s = 0.0003, 0.2
        sum_return = -9999
        print(update, comm_rank, random_w, num_g)
        x_list = collections.deque(maxlen=300)
        y_list = collections.deque(maxlen=300)

        for g in range(num_g):
            # env.reset()
            t_m = time.time()
            obs, returns, masks, actions, values, rewards, neglogpacs, states, epinfos, ret = runner.run(int(nsteps), is_test=False) #pylint: disable=E0632
            advs_ori = returns - values
            mean_returns = np.mean(returns, axis = 0)
            mean_values = np.mean(values, axis = 0)

            # sum_return = np.maximum(sum_return, mean_values[-1])

            # mean_return_gather = MPI.COMM_WORLD.gather(mean_returns)
            if comm_rank == 0:             
                print(g, comm_rank, '    ', np.array_str(mean_returns, precision=3, suppress_small=True), np.sum(np.multiply(mean_returns, random_w)))
                print(update, comm_rank, '    ', np.array_str(mean_values, precision=3, suppress_small=True))

            advs = compute_advs(advs_ori, random_w)

            if g == 0:
                init_values = np.sum(values[-1])

            if g < num_g:
                # lr_p, cr_p = 0.001, 0.5
                # print(g, comm_rank,' ***** 2 update base policy using --w', random_w, ' --alpha: lr', lr_p, ' --clip range', cr_p, ' --minibatch', nbatch_train, ' --epoch', noptepochs)
                mblossvals = nimibatch_update(  nbatch, noptepochs, nbatch_train,
                                                obs, returns, masks, actions, values, advs, neglogpacs,
                                                lr_p, cr_p, states, nsteps, model, update_type = 'all')   
                v_loss = mblossvals[1]  


        obs_gather = MPI.COMM_WORLD.gather(obs)
        returns_gather = MPI.COMM_WORLD.gather(returns)
        actions_gather = MPI.COMM_WORLD.gather(actions)
        advs_gather = MPI.COMM_WORLD.gather(advs)
        neglogpacs_gather = MPI.COMM_WORLD.gather(neglogpacs)
        sum_return_gather = MPI.COMM_WORLD.gather(mean_values[-1])
        # sum_return_gather = np.asarray(sum_return_gather)

        ws = MPI.COMM_WORLD.gather(random_w)


        if comm_rank == 0:
            for w, v in zip(ws, sum_return_gather):
                x_list.append(w)
                y_list.append(v)

            print(y_list)
            kernel = gp.kernels.Matern()
            gp_model = gp.GaussianProcessRegressor(kernel=kernel,
                                                alpha=1e-5,
                                                n_restarts_optimizer=10,
                                                normalize_y=True)

            # xp = np.array(x_list)
            # yp = np.array(y_list)
            # gp_model.fit(xp, yp)

            # suggested_w = sample_next_hyperparameter(expected_improvement, gp_model, yp, greater_is_better=True, bounds=bounds, n_restarts=100)

            n_iter = 5000

            v_pre = mean_values[-1]
            for i in range(n_iter):
                xp = np.array(x_list)
                yp = np.array(y_list)
                # Update Gaussian process with existing samples
                # model.replace_params(params_all_base, params_type='all')  
                gp_model.fit(xp, yp)

                # if i%10 == 0:
                next_sample = sample_next_hyperparameter(expected_improvement, gp_model, yp, greater_is_better=True, bounds=bounds, n_restarts=100)
                print('x next', next_sample)
                # if i == 0 or np.any(np.abs(next_sample - xp) <= 1e-7):
                #     next_sample = np.random.uniform(bounds[:, 0], bounds[:, 1], bounds.shape[0])
                #     print('random')

                for r in range(REWARD_NUM):
                    model.write_summary('weight/w_'+str(r+1), next_sample[r], i)

                y_test, sigma = gp_model.predict(next_sample.reshape(1, -1), return_std=True)

                # Obtain next noisy sample from the objective function
                obs, returns, masks, actions, values, rewards, neglogpacs, states, epinfos, ret = runner.run(int(nsteps), is_test=False) #pylint: disable=E0632
                advs_ori = returns - values
                advs = compute_advs(advs_ori, next_sample)
                # advs = (advs - advs.mean()) / (advs.std() + 1e-8)
                mean_values = np.mean(values, axis = 0)
                print(mean_values, next_sample)

                mblossvals = nimibatch_update(  nbatch, noptepochs, nbatch_train,
                                                obs, returns, masks, actions, values, advs, neglogpacs,
                                                lr_s, cr_s, states, nsteps, model, update_type = 'all')   
                display_updated_result( mblossvals, i, log_interval, nsteps, nbatch, 
                                                    rewards, returns, advs, epinfos, model, logger)                                                 
                y_real = mean_values[-1] - v_pre
                v_pre = mean_values[-1]
                
                print('y predict, y real', y_test, y_real, sigma, len(y_list))

                x_list.append(next_sample)
                y_list.append(y_real)

                # if i < 100:
                #     model.replace_params(params_all_base, params_type='actor')  

            # # save_model(model, 1)
            # print(time.time() - t_b)
            # print(update, comm_rank, '     use w', suggested_w)

            # # # compute weight for each task
            # # # sum_return_gather = sum_return_gather - sum_return_gather.min()
            # # task_weights = sum_return_gather/np.sum(sum_return_gather)
            # # max_index = np.argmax(sum_return_gather)
            # # if max_index != 0:
            # #     best_w = ws[max_index]
            # # print(sum_return_gather, task_weights, np.argmax(sum_return_gather))
            # # print(best_w)

            # model.replace_params(params_all_base, params_type='all')  
            # lr_s, cr_s = 0.0003, 0.2
            # for _ in range(10):
            #     obs, returns, masks, actions, values, rewards, neglogpacs, states, epinfos, ret = runner.run(int(nsteps), is_test=False) #pylint: disable=E0632
            #     advs_ori = returns - values
            #     advs = compute_advs(advs_ori, suggested_w)
            #     # advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            #     mean_values = np.mean(values, axis = 0)
            #     print(mean_values)
            #     mblossvals = nimibatch_update(  nbatch, noptepochs, nbatch_train,
            #                                     obs, returns, masks, actions, values, advs, neglogpacs,
            #                                     lr_s, cr_s, states, nsteps, model, update_type = 'all')   
            #     display_updated_result( mblossvals, update, log_interval, nsteps, nbatch, 
            #                                         rewards, returns, advs, epinfos, model, logger) 

            # params_all_base = model.get_current_params(params_type='all')


            # mean_returns_t = np.mean(returns, axis = 0)
            # print('after updated    ', np.array_str(mean_returns_t, precision=3, suppress_small=True), np.sum(np.multiply(mean_returns_t, random_w)))
            # for r in range(REWARD_NUM):
            #     model.write_summary('meta/mean_ret_'+str(r+1), np.mean(returns[:,r]), update)


            print('\n') 

            if save_interval and (update % save_interval == 0 or update == 1 or update == nupdates) and logger.get_dir():
                save_model(model, update)
                save_model(model, 0)
                np.save('../model/checkpoints/pf_'+str(update)+'.npy', pf_sets)
                np.save('../model/checkpoints/pf_convex.npy', pf_sets_convex)

        params_all_base = comm.bcast(params_all_base, root=0)
        first = False
    env.close()

