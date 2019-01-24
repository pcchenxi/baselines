import joblib
import time, os
import os.path as osp
import numpy as np
import tensorflow as tf

from mpi4py import MPI

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
nprocs = comm.Get_size()

REWARD_NUM = 1

class Model(object):
    def build_restore_param(self, params):
        params_all = []
        for param in params:
            layer_params = tf.placeholder(tf.float32, param.shape)
            params_all.append(layer_params)
        restores_all = []
        for p, loaded_p in zip(params, params_all):
            restores_all.append(p.assign(loaded_p))
        return params_all, restores_all

    def __init__(self, *, policy, ob_space, ac_space, goal_space, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, max_grad_norm, model_name, lr, need_summary = False):
        
        reward_size = [None, REWARD_NUM]
        # reward_size = [None]
        if need_summary:
            use_gpu = 1
        else:
            use_gpu = 0
        use_gpu = 0
        config = tf.ConfigProto(
            inter_op_parallelism_threads=1,
            intra_op_parallelism_threads=1,
            log_device_placement=False,
            device_count = { "GPU": use_gpu } )
        sess = tf.Session(config=config)

        # sess = tf.get_default_session(config=config)

        train_model = policy(sess, ob_space, ac_space, goal_space, REWARD_NUM, reuse=False, model_name = model_name)

        # A = train_model.pdtype.sample_placeholder([None])

        ADV = tf.placeholder(tf.float32, [None], name='ph_adv')
        REWARD = tf.placeholder(tf.float32, [None,1], name='ph_Rew')
        RETRUN = tf.placeholder(tf.float32, [None, 1], name='ph_R')
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None], name='ph_old_neglogpac')
        OLDVPRED = tf.placeholder(tf.float32, [None, 1], name='ph_oldvpred')
        LR = tf.placeholder(tf.float32, [], name='ph_lr')
        CLIPRANGE = tf.placeholder(tf.float32, [], name='ph_cliprange')
        STATE_F = tf.placeholder(tf.float32, [None, 64], name='fixed_state_f')

        neglogpac = train_model.pd.neglogp(train_model.A)
        entropy = tf.reduce_mean(train_model.pd.entropy())

        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        pg_losses = -ADV * ratio
        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        pg_losses3 = tf.maximum(pg_losses, pg_losses2)

        # pg_mean, pg_var = tf.nn.moments(pg_losses3, [0])
        # pg_loss_normalized = (pg_losses3 - pg_mean)/tf.sqrt(pg_var)
        # pg_loss = tf.reduce_mean(pg_loss_normalized)

        pg_loss = tf.reduce_mean(pg_losses3)

        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))

        
        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        vf_losses1 = tf.square(vpred - RETRUN)
        vf_losses2 = tf.square(vpredclipped - RETRUN)
        vf_loss = tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        # vf_loss = tf.reduce_mean(vf_losses1)

        # pg_loss1_off = -ADV * tf.clip_by_value(ratio, 0, 10)
        # adv_mean_off, adv_std_off = tf.nn.moments(pg_loss1_off, [0])
        # pg_loss_off_normalized = (pg_loss1_off - adv_mean_off)/(adv_std_off + 1e-8)
        # pg_loss_off = tf.reduce_mean(pg_loss_off_normalized)

        # vf_losses1_off = tf.reduce_sum(vf_losses1, axis = 1) * tf.clip_by_value(ratio, 0, 10)
        # vf_loss_off = tf.reduce_mean(vf_losses1_off)

        loss_a = pg_loss - entropy * ent_coef
        loss_v = vf_loss

        # loss_a_off = pg_loss_off
        # loss_v_off = vf_loss_off

        # loss_forward = tf.reduce_mean(tf.reduce_sum(tf.square(train_model.x_next_feature_pred - STATE_F), 1))
        loss_forward = tf.reduce_mean(tf.reduce_sum(tf.square(train_model.x_next_feature_pred - train_model.x_next_feature), 1))
        # error_rew = tf.square(train_model.p_error-REWARD)
        # loss_forward = tf.reduce_mean(error_rew)

        eir_diff = tf.square(train_model.expected_ir - REWARD)
        loss_eir = tf.reduce_mean(eir_diff)

        self.loss_names = ['policy_loss', 'value_loss', 'forward', 'policy_entropy', 'approxkl', 'clipfrac', 'grad_norm']

        with tf.variable_scope(model_name):
            params = tf.global_variables(scope=model_name)
            params_actor = tf.global_variables(scope=model_name+'/actor')
            # params_logstd = tf.global_variables(scope=model_name+'/actor/logstd')
            params_value = tf.global_variables(scope=model_name+'/value')
            params_forward = tf.global_variables(scope=model_name+'/forward_dynamic')
            params_forward_train = tf.global_variables(scope=model_name+'/forward_dynamic/pred_state_feature')
            params_expected_ir = tf.global_variables(scope=model_name+'/forward_dynamic/expected_ir')


        grads_a = tf.gradients(loss_a, params_actor)
        # grads_a_off = tf.gradients(loss_a_off, params_actor)
        max_grad_norm = 1
        if max_grad_norm is not None:
            grads_a, _grad_norm_a = tf.clip_by_global_norm(grads_a, max_grad_norm)
            # grads_a_off, _grad_norm_a = tf.clip_by_global_norm(grads_a_off, max_grad_norm)
        grads_a = list(zip(grads_a, params_actor))
        # grads_a_off = list(zip(grads_a_off, params_actor))


        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            optimizer_a = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
            train_a = optimizer_a.apply_gradients(grads_a)

            # optimizer_a_off = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
            # train_a_off = optimizer_a_off.apply_gradients(grads_a_off)

            # optimizer_a = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
            # train_a = optimizer_a.minimize(loss_a, var_list=params_actor)

            # optimizer_v_off = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
            # train_v_off = optimizer_v_off.minimize(loss_v_off, var_list=params_value)

            optimizer_v = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
            train_v = optimizer_v.minimize(loss_v, var_list=params_value)

            optimizer_forward = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
            train_forward = optimizer_forward.minimize(loss_forward, var_list=params_forward_train)

            # optimizer_a_off = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
            # train_a_off = optimizer_a_off.minimize(loss_a_off, var_list=params_actor)

            optimizer_ir = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
            train_ir = optimizer_ir.minimize(loss_eir, var_list=params_expected_ir)

        ############################################################################################
        network_params_all, restores_all = self.build_restore_param(params)
        network_params_actor, restores_actor = self.build_restore_param(params_actor)
        network_params_value, restores_value = self.build_restore_param(params_value)
        network_params_fm, restores_fm = self.build_restore_param(params_forward)
        network_params_eir, restores_eir = self.build_restore_param(params_expected_ir)
        ############################################################################################


        def train(lr, cliprange, obs, obs_next, returns, actions, values, advs, rewards, fixed_state_f, neglogpacs, train_type='all', use_off_policy=False):
            # advs_suggest = advs_suggest / (advs_suggest.std() + 1e-8)
            # obs_ret = np.concatenate((obs, returns[:, 0:-1]), axis=1)
            # obs_ret = np.concatenate((obs, actions), axis=1)

            advs = (advs - advs.mean()) / (advs.std() + 1e-8)                
            td_map = {train_model.is_training: True, train_model.X:obs, train_model.X_NEXT:obs_next, train_model.A:actions, ADV:advs, RETRUN:returns.reshape(-1,1), 
                    REWARD:rewards.reshape(-1,1), STATE_F:fixed_state_f, LR:lr, CLIPRANGE:cliprange, OLDNEGLOGPAC:neglogpacs, OLDVPRED:values.reshape(-1,1), }

            if train_type == 'all':
                if use_off_policy: 
                    return sess.run(
                        [pg_loss, loss_v, loss_forward, entropy, approxkl, clipfrac, _grad_norm_a, train_a_off, train_v_off, train_forward],
                        td_map                
                    )[:-3]
                else:          
                    return sess.run(
                        [pg_loss, loss_v, loss_forward, entropy, approxkl, clipfrac, _grad_norm_a, train_a, train_v, train_forward],
                        td_map                
                    )[:-3]
            if train_type == 'av':             
                # mean, std = sess.run([adv_mean, adv_std], td_map)
                # pgl3, pg = sess.run([pg_losses3, pg_loss_normalized], td_map)
                # pgl4 = (pgl3 - pgl3.mean()) / (pgl3.std())
                # for p3, p, p4 in zip(pgl3, pg, pgl4):
                #     print(p3, p, p4)
                # print(mean, std, pgl3.mean(), pgl3.var())

                return sess.run(
                    [pg_loss, loss_v, loss_forward, entropy, approxkl, clipfrac, _grad_norm_a, train_a, train_v],
                    td_map                
                )[:-2]
            if train_type == 'pm':
                return sess.run(
                    [pg_loss, loss_v, loss_forward, entropy, approxkl, clipfrac, _grad_norm_a, train_forward],
                    td_map                
                )[:-1]


        def train_pm(lr, obs, a, obs_, fixed_state_f, rewards, train_type):
            # print(obs.shape)

            td_map = {train_model.is_training: True, train_model.X:obs, train_model.A:a, train_model.X_NEXT:obs_, STATE_F:fixed_state_f, REWARD:rewards.reshape(-1,1), LR:lr}
            if train_type == 'pm':
                return sess.run([loss_forward, train_forward], td_map)[:-1]
            if train_type == 'trace':
                return sess.run([loss_eir, train_ir], td_map)[:-1]

        def get_neglogpac(obs, actions):
            return sess.run(neglogpac, {train_model.X:obs, train_model.A:actions})

        def get_actions_dist(obs):
            return sess.run([train_model.mean, train_model.std], {train_model.X:obs})

        def save(save_path, save_path_base):
            # saver.save(sess, save_path + '.cptk') 
            ps = sess.run(params)
            joblib.dump(ps, save_path)
            joblib.dump(ps, save_path_base)
            print('save model', save_path_base)

        def load(load_path, load_value = True):
            loaded_params = joblib.load(load_path)
            # # print('parama', len(loaded_params), load_path)
            # restores = []
            # for p, loaded_p in zip(params, loaded_params):
            #     find_vf = p.name.find('vf')
            #     # if find_vf != -1:
            #     #     print(p.name, loaded_p)
            #     if load_value == False and find_vf != -1:
            #         continue
            #     restores.append(p.assign(loaded_p))
            # sess.run(restores)
            sess.run(restores_all, feed_dict={i: d for i, d in zip(network_params_all, loaded_params)})
            # print('loaded')

        def load_and_increase_logstd(load_path, increase_amount = 0.2):
            loaded_params = joblib.load(load_path)
            sess.run(restores_all, feed_dict={i: d for i, d in zip(network_params_all, loaded_params)})

            # param_actor = sess.run(params_actor)
            # sess.run(restores_actor, feed_dict={i: d + (np.random.rand(*np.asarray(d).shape)-0.5)*0.3 for i, d in zip(network_params_actor, param_actor)})

            param_logstd = sess.run(params_logstd)
            sess.run(restores_logstd, feed_dict={i: d+np.random.rand()*0.3 for i, d in zip(network_params_logstd, param_logstd)})

            #     if load_value == False and find_vf != -1:
            #         continue
            #     restores.append(p.assign(loaded_p))
            # sess.run(restores)
            # saver.restore(sess, load_path + '.cptk')
            # print('loaded')
            # If you want to load weights, also save/load observation scaling inside VecNormalize

        def get_current_params(params_type = 'all'):
            if params_type == 'all':
                net_params = sess.run(params)
            elif params_type == 'actor':
                net_params = sess.run(params_actor)
            elif params_type == 'value':
                net_params = sess.run(params_value)
            elif params_type == 'forward':
                net_params = sess.run(params_forward)
            elif params_type == 'eir':
                net_params = sess.run(params_expected_ir)                        
            return np.asarray(net_params)

        def replace_params(loaded_params, params_type = 'all'):
            if params_type == 'all':
                sess.run(restores_all, feed_dict={i: d for i, d in zip(network_params_all, loaded_params)})
            elif params_type == 'actor':
                sess.run(restores_actor, feed_dict={i: d for i, d in zip(network_params_actor, loaded_params)})
            elif params_type == 'value':
                sess.run(restores_value, feed_dict={i: d for i, d in zip(network_params_value, loaded_params)})
            elif params_type == 'forward':
                sess.run(restores_fm, feed_dict={i: d for i, d in zip(network_params_fm, loaded_params)})
            elif params_type == 'eir':
                sess.run(restores_eir, feed_dict={i: d for i, d in zip(network_params_eir, loaded_params)})

        def write_summary(summary_name, value, step):
            summary = tf.Summary()
            summary.value.add(tag=summary_name, simple_value=float(value))
            self.summary_writer.add_summary(summary, step)
            self.summary_writer.flush()  

        def log_histogram(tag, values, step, bins=1000):
            """Logs the histogram of a list/vector of values."""
            # Convert to a numpy array
            values = np.array(values)
            
            # Create histogram using numpy        
            counts, bin_edges = np.histogram(values, bins=bins)

            # Fill fields of histogram proto
            hist = tf.HistogramProto()
            hist.min = float(np.min(values))
            hist.max = float(np.max(values))
            hist.num = int(np.prod(values.shape))
            hist.sum = float(np.sum(values))
            hist.sum_squares = float(np.sum(values**2))

            # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
            # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
            # Thus, we drop the start of the first bin
            bin_edges = bin_edges[1:]

            # Add bin edges and counts
            for edge in bin_edges:
                hist.bucket_limit.append(edge)
            for c in counts:
                hist.bucket.append(c)

            # Create and write Summary
            summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
            self.summary_writer.add_summary(summary, step)
            self.summary_writer.flush()

        self.train = train
        self.train_pm = train_pm
        self.train_model = train_model
        # self.act_model = act_model
        self.step = train_model.step
        self.step_max = train_model.step_max
        self.value = train_model.value
        self.get_neglogpac = get_neglogpac
        self.get_actions_dist = get_actions_dist

        self.save = save
        self.load = load
        self.load_and_increase_logstd = load_and_increase_logstd
        self.replace_params = replace_params
        self.get_current_params = get_current_params
        self.get_expected_ir = train_model.get_expected_ir

        self.state_action_pred = train_model.state_action_pred
        self.state_feature = train_model.state_feature
        self.pred_error = train_model.pred_error

        if need_summary:
            # self.summary_writer = tf.summary.FileWriter('../model/log/exp/puck/normal', sess.graph)   
            # log_path = '../model/log/exp/landing/mp_morl_meta' #+ str(comm_rank)
            # log_path = '../model/log/exp/ant/meta_finetune_' + str(comm_rank)
            # log_path = '../model/log/exp/cheetah/mp_train_' + str(comm_rank)
            log_path = '../model/log/exp/puck/normal'

            os.makedirs(log_path, exist_ok=True)
            # self.summary_writer = tf.summary.FileWriter('../model/log/human_hard')   
            self.summary_writer = tf.summary.FileWriter(log_path)  
            self.write_summary = write_summary
            self.log_histogram = log_histogram
        
        saver = tf.train.Saver()

        tf.global_variables_initializer().run(session=sess) #pylint: disable=E1101


class Runner(object):

    def __init__(self, *, env, model, nsteps, gamma, lam, buffer_length = 200):
        self.env = env
        self.model = model
        nenv = 1 #env.num_envs
        self.nenv = nenv
        self.obs = np.zeros([1, 28], dtype=model.train_model.X.dtype.name)
        # self.obs[:] = env.reset()
        self.gamma = np.array([gamma, gamma])
        self.lam = np.array([lam, 1])
        self.nsteps = nsteps
        self.dones = [False for _ in range(nenv)]

        self.full_obs = self.env.reset()

    def pm_activate(self, pm, scale):
        pm_scaled = pm/scale
        pm_act = np.expm1(pm_scaled)/5
        return np.clip(pm_act, 0, 1)


    def compute_intrinsic_reward(self, obs, a, obs_):
        return self.model.pred_error([obs], [a], [obs_])[0]


    def run(self, model, nsteps, pm_cut, pm_scale, is_test = False, use_deterministic = False, render = False, random_prob = 0.5):
        self.model = model

        mb_obs, mb_obs_next, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs, mb_boost = [],[],[],[],[],[],[],[]
        mb_infos, epinfos, mb_fixed_state_f = [], [], []
        mb_pm_state = []

        full_obs = self.full_obs.copy()
        obs = np.concatenate((full_obs['observation'], full_obs['desired_goal']), axis=0) 

        batch_rew = []
        step_count = 0
        for t in range(nsteps):            
            if use_deterministic:
                actions, values, neglogpacs = self.model.step_max([obs], self.dones)
            else:
                actions, values, neglogpacs = self.model.step([obs], self.dones)

            neglogpacs = self.model.get_neglogpac([obs], actions)
            # print(actions)
            self.full_obs = full_obs.copy()
            mb_obs.append(obs.copy())
            mb_actions.append(actions[0])
            mb_values.append(values[0])
            mb_neglogpacs.append(neglogpacs[0])
            mb_dones.append(self.dones)
            mb_boost.append(np.array([0.0, 0.0]))
            # next_statef_pred = self.model.state_action_pred([obs], actions)

            full_obs, r, self.dones, infos = self.env.step(actions[0])
                
            obs = np.concatenate((full_obs['observation'], full_obs['desired_goal']), axis=0) 

            mb_obs_next.append(obs.copy())

            fixed_state_f = self.model.state_feature([mb_obs[-1]], [mb_actions[-1]], [mb_obs_next[-1]])[0]
            mb_fixed_state_f.append(fixed_state_f)

            # ##########################################################
            ## for roboschool
            if render:
                self.env.render('human')
                # print(values)
            
            if self.dones or full_obs['achieved_goal'][-1] < 0.35 or step_count > 300 or t == nsteps-1:
                self.dones = True
                #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! only for this setting!!!!!!!!!!!!
                #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! only for this setting!!!!!!!!!!!!
                #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! only for this setting!!!!!!!!!!!!
                if full_obs['achieved_goal'][-1] < 0.35:
                    v_preds = values[0] #self.model.value([obs])[0]
                else:
                    v_preds = self.model.value([obs])[0]

                mb_boost[-1][0] = (self.gamma*v_preds)[0]
                # mb_boost[-1] = (self.gamma*v_preds)

                full_obs = self.env.reset()   
                obs = np.concatenate((full_obs['observation'], full_obs['desired_goal']), axis=0) 
                self.full_obs = full_obs.copy()
            
            r_ext = (r+1)*1
            r_pm = self.compute_intrinsic_reward(mb_obs[-1], mb_actions[-1], mb_obs_next[-1])
            r_pm_act = self.pm_activate(r_pm, pm_scale) + r_ext
            r_pm_crop = np.maximum(0, r_pm_act - pm_cut)

            r_potental = self.model.value([mb_obs_next[-1]])[0][-1] - values[0][-1]

            r_ints = r_pm_act + r_potental*1 

            rewards = np.array([r_ints, r_pm_crop])

            # rewards[1] = r_pm + r +1 #self.model.get_self_reward([obs])[0]
            mb_rewards.append(rewards)
            batch_rew.append([r, r_pm, r_pm_act, r_potental])

            step_count += 1

            if self.dones:
                step_count = 0

            if is_test and (step_count > 50 or self.dones):
                break
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_obs_next = np.asarray(mb_obs_next, dtype=self.obs.dtype)
        mb_fixed_state_f = np.asarray(mb_fixed_state_f, dtype=self.obs.dtype)
        # mb_pm_state = np.asarray(mb_pm_state, dtype=self.obs.dtype)

        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_boost = np.asarray(mb_boost, dtype=np.float32)
        batch_rew = np.asarray(batch_rew)

        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value([obs], self.dones)[0]

        #discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0

        for t in reversed(range(len(mb_values))):
            if t == len(mb_values) - 1:
                nextnonterminal = 0
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]     
                nextvalues = mb_values[t+1]

            delta = mb_rewards[t] + mb_boost[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]

            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam

        mb_returns = mb_advs + mb_values

        # print(mb_values.shape, mb_rewards.shape, mb_returns.shape)
        if render:
            print(self.gamma, self.lam)
            with np.printoptions(precision=3, suppress=True):
                for i in range(len(mb_values)):
                    print(batch_rew[i], np.asarray(mb_rewards[i]), np.asarray(mb_returns[i]), mb_advs[i][0], mb_values[i])

            print(self.pm_cut, self.pm_scale)
        
        # for ret, rew in zip(mb_returns, mb_rewards):
        #     print(ret, rew)
        mb_rewards = np.concatenate((batch_rew[:,0].reshape(-1, 1), mb_rewards), axis=1)
        return (mb_obs, mb_obs_next, mb_returns, mb_dones, mb_actions, mb_values, mb_advs, mb_rewards, mb_fixed_state_f, mb_pm_state, mb_neglogpacs, epinfos)
        # return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_advs, mb_rewards, mb_neglogpacs)),
        #     epinfos, ret)

def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

def constfn(val):
    def f(_):
        return val
    return f

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

def save_model(model, model_name):
    checkdir = osp.join('../model', 'checkpoints')
    os.makedirs(checkdir, exist_ok=True)
    savepath_base = osp.join(checkdir, str(model_name))
    # print('    Saving to', savepath_base, savepath_base)
    model.save(savepath_base, savepath_base)

def display_updated_result( lossvals, update, log_interval, nsteps, nbatch, 
                            rewards, returns, values, advs_ori, epinfobuf, model, logger):
    # lossvals = np.mean(mblossvals, axis=0)
    if update % log_interval == 0 or update == 1:
        step_label = update
        for i in range(2):
            model.write_summary('return/mean_ret_'+str(i+1), np.mean(returns[:,i]), step_label)
            # model.write_summary('sum_rewards/rewards_'+str(i+1), safemean([epinfo['r'][i] for epinfo in epinfobuf]), step_label)
            # model.log_histogram('return_dist/return_'+str(i+1), returns[:,i], step_label)
        for i in range(3):
            model.write_summary('reward/mean_rew_'+str(i+1), np.mean(rewards[:,i]), step_label)
            
            
        model.write_summary('sum_rewards/score', safemean([epinfo['r'][-1] for epinfo in epinfobuf]), step_label)
        model.write_summary('sum_rewards/mean_length', safemean([epinfo['l'] for epinfo in epinfobuf]), step_label)

        for (lossval, lossname) in zip(lossvals, model.loss_names):
            model.write_summary('loss/'+lossname, lossval, step_label)


def nimibatch_update(   nbatch, noptepochs, nbatch_train,
                        obs, obs_next, returns, actions, values, advs, rewards, fixed_state_f, neglogpacs,
                        lrnow, cliprangenow, model, update_type = 'all', allow_early_stop = True, use_off_policy=False):
    mblossvals = []
    nbatch_train = int(nbatch_train)

    # rewards = rewards[:,0]
    # inds = np.arange(nbatch)
    inds = np.arange(len(obs))
    early_stop = False
    for ep in range(noptepochs):
        np.random.shuffle(inds)
        for start in range(0, len(obs), nbatch_train):
            end = start + nbatch_train
            if end >= len(obs):
                break
            mbinds = inds[start:end]
            slices = (arr[mbinds] for arr in (obs, obs_next, returns[:,0], actions, values[:,0], advs, rewards, fixed_state_f, neglogpacs))

            losses = model.train(lrnow, cliprangenow, *slices, train_type=update_type, use_off_policy=use_off_policy)
            if losses != []:
                mblossvals.append(losses)

    lossvals = np.mean(mblossvals, axis=0)
    return lossvals

def nimibatch_update_pm(noptepochs, nbatch_train, pm_states, rewards, lrnow, model, update, mode):
    mblossvals = []
    nbatch_train = int(nbatch_train)

    # pm_states = np.asarray(pm_states)
    obs = pm_states[0]
    a = pm_states[1]
    obs_ = pm_states[2]
    fixed_state_f = pm_states[3]
    # rewards = rewards[:,0]

    if mode == 'pm':
        inds = np.random.choice(len(obs), 30000)
    else:
        # prob = rewards - rewards.min()
        prob = rewards/rewards.sum()
        # print(prob)
        inds = np.random.choice(len(obs), 30000, p=rewards/rewards.sum())

    for ep in range(noptepochs):
        np.random.shuffle(inds)
        # for start in range(0, len(obs), nbatch_train):
        for start in range(0, len(inds), nbatch_train):
            end = start + nbatch_train
            if end >= len(obs):
                break

            mbinds = inds[start:end]
            slices = (arr[mbinds] for arr in (obs, a, obs_, fixed_state_f, rewards))

            losses = model.train_pm(lrnow, *slices, mode)
            if losses != []:
                mblossvals.append(losses)

    lossvals = np.mean(mblossvals, axis=0)
    if mode == 'pm':
        model.write_summary('loss/forward', lossvals[0], update)
    elif mode == 'trace':
        model.write_summary('loss/trace', lossvals[0], update)

def compute_advs( advs_ori, dir_w):

    # advs_ori = (advs_ori - np.mean(advs_ori, axis=0)) / (np.std(advs_ori, axis=0) + 1e-8)

    dir_w = np.asarray(dir_w)
    dir_w_length = np.sqrt(np.sum(dir_w*dir_w))
    advs = np.zeros(len(advs_ori))
    # advs_nrom = np.zeros_like(advs_ori)
    # advs_nrom = (advs_ori - np.mean(advs_ori, axis=0)) / (np.std(advs_ori, axis=0) + 1e-8)
    # advs_nrom = advs_ori / (np.std(advs_ori, axis=0) + 1e-8)
    # advs_max = np.abs(np.max(advs_ori, axis=0))

    for t in range(len(advs_ori)):
        advs[t] = np.sum(np.multiply(advs_ori[t], dir_w))/dir_w_length

        # print (np.array_str(advs_ori[t], precision=3, suppress_small=True) \
        # , np.array_str(advs_nrom[t], precision=3, suppress_small=True) \
        # , advs[t]
        # )

    advs = (advs - advs.mean()) / (advs.std() + 1e-8)
    # advs = np.clip(advs, -5, 5)
    # for t in range(len(advs_ori)):
    #     if advs[t] < 0:
    #         advs[t] = -1
    #     elif advs[t] > 0:
    #         advs[t] = 1
    # advs = advs / (advs.std() + 1e-8)

    # for t in range(len(advs_ori)):
    #     print (np.array_str(advs_ori[t], precision=3, suppress_small=True) \
    #     , np.array_str(advs_nrom[t], precision=3, suppress_small=True) \
    #     , advs[t]
    #     )

    return advs

def compute_advs_singleobj(values, returns):
    returns_single = np.sum(returns, axis = 1)
    values_single = np.sum(values, axis = 1)

    advs = returns_single - values_single

    return advs
