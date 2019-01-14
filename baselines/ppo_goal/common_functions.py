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
        # REWARD = tf.placeholder(tf.float32, [None, REWARD_NUM-1], name='ph_Rew')
        RETRUN = tf.placeholder(tf.float32, [None, 2], name='ph_R')
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None], name='ph_old_neglogpac')
        OLDVPRED = tf.placeholder(tf.float32, [None, 2], name='ph_oldvpred')
        LR = tf.placeholder(tf.float32, [], name='ph_lr')
        CLIPRANGE = tf.placeholder(tf.float32, [], name='ph_cliprange')
        STATE_F = tf.placeholder(tf.float32, [None, 128], name='fixed_state_f')

        neglogpac = train_model.pd.neglogp(train_model.A)
        entropy = tf.reduce_mean(train_model.pd.entropy())

        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        pg_losses = -ADV * ratio
        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        # pg_loss = tf.reduce_mean(pg_losses2)
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))

        
        vpred = train_model.vf
        # vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        vf_losses1 = tf.square(vpred - RETRUN)
        vf_losses1 = tf.reduce_sum(vf_losses1, axis = 1)*ratio
        # vf_losses2 = tf.square(vpredclipped - RETRUN)
        # vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        vf_loss = tf.reduce_mean(vf_losses1)


        loss_a = pg_loss - entropy * ent_coef
        loss_v = vf_loss

        # loss_forward = tf.reduce_mean(tf.reduce_sum(tf.square(train_model.x_next_feature_pred - STATE_F), 1))
        loss_forward = tf.reduce_mean(tf.reduce_sum(tf.square(train_model.x_next_feature_pred - train_model.x_next_feature), 1))

        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac', 'grad_norm']

        with tf.variable_scope(model_name):
            params = tf.global_variables(scope=model_name)
            params_actor = tf.global_variables(scope=model_name+'/actor')
            params_logstd = tf.global_variables(scope=model_name+'/actor/logstd')
            params_value = tf.global_variables(scope=model_name+'/value')
            params_forward = tf.global_variables(scope=model_name+'/forward_dynamic')


        grads_a = tf.gradients(loss_a, params_actor)
        max_grad_norm = 1
        if max_grad_norm is not None:
            grads_a, _grad_norm_a = tf.clip_by_global_norm(grads_a, max_grad_norm)
        grads_a = list(zip(grads_a, params_actor))


        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            optimizer_a = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
            train_a = optimizer_a.apply_gradients(grads_a)


            # optimizer_a = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
            # train_a = optimizer_a.minimize(loss_a, var_list=params_actor)

            optimizer_v = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
            train_v = optimizer_v.minimize(loss_v, var_list=params_value)

            optimizer_forward = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
            train_forward = optimizer_forward.minimize(loss_forward, var_list=params_forward)
        ############################################################################################
        network_params_all = []
        for param in params:
            layer_params = tf.placeholder(tf.float32, param.shape)
            network_params_all.append(layer_params)
        restores_all = []
        for p, loaded_p in zip(params, network_params_all):
            restores_all.append(p.assign(loaded_p))

        network_params_actor = []
        for param in params_actor:
            layer_params = tf.placeholder(tf.float32, param.shape)
            network_params_actor.append(layer_params)
        restores_actor = []
        for p, loaded_p in zip(params_actor, network_params_actor):
            restores_actor.append(p.assign(loaded_p))

        network_params_logstd = []
        for param in params_logstd:
            layer_params = tf.placeholder(tf.float32, param.shape)
            network_params_logstd.append(layer_params)
        restores_logstd = []
        for p, loaded_p in zip(params_logstd, network_params_logstd):
            restores_logstd.append(p.assign(loaded_p))

        network_params_value = []
        for param in params_value:
            layer_params = tf.placeholder(tf.float32, param.shape)
            network_params_value.append(layer_params)
        restores_value = []
        for p, loaded_p in zip(params_value, network_params_value):
            restores_value.append(p.assign(loaded_p)) 

        network_params_fm = []
        for param in params_forward:
            layer_params = tf.placeholder(tf.float32, param.shape)
            network_params_fm.append(layer_params)
        restores_fm = []
        for p, loaded_p in zip(params_forward, network_params_fm):
            restores_fm.append(p.assign(loaded_p)) 
        ############################################################################################


        def train(lr, cliprange, obs, obs_next, returns, actions, values, advs, rewards, fixed_state_f, neglogpacs, train_type='all'):
            # advs_suggest = advs_suggest / (advs_suggest.std() + 1e-8)
            # obs_ret = np.concatenate((obs, returns[:, 0:-1]), axis=1)
            obs_ret = np.concatenate((obs, actions), axis=1)

            advs = (advs - advs.mean()) / (advs.std() + 1e-8)                
            td_map = {train_model.is_training: True, train_model.X:obs, train_model.X_NEXT:obs_next, train_model.A:actions, ADV:advs, RETRUN:returns, 
                    STATE_F:fixed_state_f, LR:lr, CLIPRANGE:cliprange, OLDNEGLOGPAC:neglogpacs, OLDVPRED:values}

            if train_type == 'all':
                # vp, ret, v_l1 = sess.run(
                #     [OLDNEGLOGPAC, neglogpac, ratio],
                #     td_map                
                # )
                # print(vp.shape, ret.shape, v_l1.shape)

                return sess.run(
                    [pg_loss, loss_v, entropy, approxkl, clipfrac, _grad_norm_a, train_a, train_v, train_forward],
                    td_map                
                )[:-3]
            if train_type == 'av':
                return sess.run(
                    [pg_loss, loss_v, entropy, approxkl, clipfrac, _grad_norm_a, train_a, train_v],
                    td_map                
                )[:-2]
            if train_type == 'pm':
                if rewards[:, -1].mean() > 0:
                    return sess.run(
                        [pg_loss, loss_v, loss_forward, entropy, approxkl, clipfrac, _grad_norm_a, train_forward],
                        td_map                
                    )[:-1]
                else:
                    print('skip', rewards[:, -1].mean())
                    return []

        def train_pm(lr, obs, obs_):
            td_map = {train_model.is_training: True, train_model.X:obs, train_model.X_NEXT:obs_, LR:lr}
            return sess.run([loss_forward, train_forward], td_map)[:-1]

        def get_neglogpac(obs, actions):
            return sess.run(neglogpac, {train_model.X:obs, A:actions})

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
        self.gamma = gamma #np.full(REWARD_NUM, gamma)
        # self.gamma[-1] = 0.98
        self.lam = lam
        self.nsteps = nsteps
        self.dones = [False for _ in range(nenv)]

        self.tasks_sas = []
        self.tasks = []
        self.tasks_tra = []
        self.tasks_init_data = []
        self.tasks_value = []
        self.min_value = 0
        self.index = 0
        self.task_length = buffer_length
        self.gamma_list = np.random.rand(nprocs) * 0.1
        self.gamma_list[0] = 0

        self.history_states = []
        self.history_states_index = 0
        self.history_buff_length = 500

        self.init_task_buffer()

    def init_task_buffer(self):
        full_obs = self.env.reset()  
        obs = np.concatenate((full_obs['observation'], full_obs['desired_goal']), axis=0) 
        self.full_obs = full_obs.copy()
        actions, values, neglogpacs = self.model.step([obs], self.dones)

        sas = [obs, actions, obs]
        task_obs = obs
        current_sim_data = self.env.get_sim_data()
        task = [current_sim_data, full_obs['desired_goal']]        
        for i in range(self.task_length):
            self.tasks.append(task)
            self.tasks_value.append(0.0)
            self.tasks_sas.append(sas) 
            self.tasks_tra.append([sas])
            self.tasks_init_data.append([])

        for i in range(self.history_buff_length):
            self.history_states.append(obs)

    def init_task_pool(self, nsteps, model_old, render = False):
        self.tasks = []
        self.tasks_value = []

        tra_obs = []
        tra_data = []

        full_obs = self.env.reset()       
        # full_obs = self.env.set_sim_data(sim_data, sim_goal)

        start_obs = full_obs['observation']
        task_data = self.env.get_sim_data()
        # sim_goal = full_obs['desired_goal']

        tra_obs.append(full_obs)
        tra_data.append(task_data)

        self.obs[:] = np.concatenate((full_obs['observation'], full_obs['desired_goal']), axis=0) 
        for t in range(nsteps):
            full_obs = self.env.reset()
            # if len(self.tasks) > 0 and np.random.rand()>0.5:
            #     task_index = np.random.randint(len(self.tasks))
            #     task = self.tasks[task_index]     
            #     full_obs = self.env.set_sim_data(task[0], task[1])        print(vf_losses1.shape)


            start_obs = full_obs['observation']
            task_data = self.env.get_sim_data() 
            goal_pre = full_obs['achieved_goal'] #np.zeros(len(full_obs['achieved_goal'])) #full_obs['achieved_goal']

            tra_tasks = []
            tra_values = []

            for s in range(150):
                actions, values, neglogpacs = self.model.step(self.obs, self.dones)
                full_obs, rewards, self.dones, infos = self.env.step(actions[0])
                self.obs[:] = np.concatenate((full_obs['observation'], full_obs['desired_goal']), axis=0) 

                # check task
                task_obs = np.concatenate((start_obs, full_obs['achieved_goal']), axis=0) 
                task_value = self.model.value([task_obs])[0,0]
                # task_value_old = model_old.value([task_obs])[0,0]
                # task = [task_data, full_obs['achieved_goal']]
                if render:
                    self.env.render('human')
                    print(task_value, values, full_obs['achieved_goal'])
                # print(task_value, goal_pre, full_obs['achieved_goal'])

                # save current state data
                current_sim_data = self.env.get_sim_data()
                tra_obs.append(full_obs)
                tra_data.append(current_sim_data)

                diff = abs(goal_pre - full_obs['achieved_goal'])
                if np.any(diff > 0.01):
                    for ob, data in zip(tra_obs, tra_data):
                        task_obs = np.concatenate((ob['observation'], full_obs['achieved_goal']), axis=0) 
                        task_value = self.model.value([task_obs])[0,0]     

                        task = [data, full_obs['achieved_goal']]
                        tra_tasks.append(task)
                        tra_values.append(task_value)      

                    task_obs = np.concatenate((full_obs['observation'], full_obs['desired_goal']), axis=0) 
                    task_value = self.model.value([task_obs])[0,0]     
                    task = [current_sim_data, full_obs['desired_goal']]
                    tra_tasks.append(task)
                    tra_values.append(task_value) 

                    if render:
                        print('add task')
                # task_value = np.clip(task_value, 0.000001, 1)

                goal_pre = full_obs['achieved_goal']

                if self.dones or full_obs['achieved_goal'][-1] < 0.4:
                    tra_obs = []
                    tra_data = []     

                    for task, task_value, i in zip(tra_tasks, tra_values, range(len(tra_tasks))):
                        if i%3 == 0:
                            self.env.reset()  
                            full_obs = self.env.set_sim_data(task[0], task[1])
                            task_obs = np.concatenate((full_obs['observation'], full_obs['desired_goal']), axis=0) 
                            real_value = self.model.value([task_obs])[0,0] 
                            real_value_old = model_old.value([task_obs])[0,0]
                            # if render:
                            #     print(real_value, real_value_old)
                            self.env.render('human')
                            # print(real_value)
                            value_diff = abs(real_value - real_value_old)
                            if real_value < 0:
                                self.tasks.append(task)
                                self.tasks_value.append(real_value)    
                    if render:
                        print('task length', len(tra_tasks), len(self.tasks))
                    break   

            if len(self.tasks) > 300:
                break

        if len(self.tasks) > 0:
            print(len(self.tasks), np.mean(self.tasks_value), np.max(self.tasks_value), np.min(self.tasks_value))      
            self.tasks_value = -np.asarray(self.tasks_value)
            mean_v = self.tasks_value.mean()
            # final_task, final_v = [], []
            # for task, v in zip(self.tasks, self.tasks_value):
            #     if v > mean_v:
            #         final_task.append(task)
            #         final_v.append(v)
            # self.tasks = final_task.copy()
            # self.tasks_value = final_v.copy()
            self.tasks_value = self.tasks_value - self.tasks_value.min()
            self.tasks_value = self.tasks_value/self.tasks_value.sum()            
            print(len(self.tasks), np.mean(self.tasks_value), np.max(self.tasks_value), np.min(self.tasks_value))      

    def update_task_value(self, task_list, sas_list, value_list=[]):
        task_values = []
        for i in range(len(task_list)):
            if value_list != [] and value_list[i] == 0:
                diff_f_norm = 0
            else:
                task = task_list[i]
                obs = sas_list[i][0]
                a = sas_list[i][1]
                obs_ = sas_list[i][2]

                if np.random.rand() > 0:
                    diff_f_norm = self.compute_intrinsic_reward(obs, a, obs_)
                else:
                    self.env.reset()  
                    full_obs = self.env.set_sim_data(task[0], task[1])
                    obs = np.concatenate((full_obs['observation'], full_obs['desired_goal']), axis=0) 
                    
                    full_obs, _, _, _ = self.env.step(a)
                    obs_ = np.concatenate((full_obs['observation'], full_obs['desired_goal']), axis=0) 

                    diff_f_norm = self.compute_intrinsic_reward(obs, a, obs_)

                v_preds = self.model.value([obs_])[0][-1]
                diff_f_norm += v_preds*self.gamma_list[comm_rank]

            task_values.append(diff_f_norm)
        return task_values

    def compute_intrinsic_reward(self, obs, a, obs_):
        # # object_related_index = [3,4,5,11,12,13,14,15,16,17,18,19]
        # next_statef = self.model.state_feature([obs], [a], [obs_])
        # next_statef_pred = self.model.state_action_pred([obs], [a], [obs_])

        # diff_f = (next_statef_pred - next_statef) #np.take((next_statef_pred - next_statef), object_related_index)
        # diff_f_norm = np.sqrt(np.sum(diff_f*diff_f))
        # return self.model.pred_error([obs], [a], [obs_])[0]

        obs_list = np.full((self.history_buff_length, len(obs_)), obs_)
        a_list = np.full((self.history_buff_length, len(a)), a)
        start_states = np.asarray(self.history_states)

        reward_list = self.model.pred_error(start_states, a_list, obs_list)
        # print(reward_list)
        return reward_list.mean()

    def reset_env_sas(self, init_data, tra_sas):
        full_obs = self.env.set_sim_data(init_data[0], init_data[1])
        print(len(tra_sas))
        for s, a, _ in tra_sas:
            full_obs, r, self.dones, infos = self.env.step(a)
            self.env.render('human')
        print('replay')
        return full_obs

    def run(self, model, nsteps, is_test = False, use_deterministic = False, render = False, random_prob = 0.5, need_pm = True):
        self.model = model
        mb_obs, mb_obs_next, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs, mb_boost = [],[],[],[],[],[],[],[]
        mb_infos, epinfos, mb_fixed_state_f = [], [], []
        mb_pm_state = []
        # self.tasks = []
        # self.tasks_value = []
        
        tra_tasks = []
        tra_tasks_value = []
        tra_tasks_sas = []
        tra_tasks_data = []
        tra_rewards = []

        # update task value
        # print('average task value before', np.asarray(self.tasks_value).mean())
        self.tasks_value = self.update_task_value(self.tasks, self.tasks_sas, value_list=self.tasks_value)
        # print('average task value after', np.asarray(self.tasks_value).mean())

        min_index = np.argmin(np.asarray(self.tasks_value))

        full_obs = self.full_obs.copy()
        obs = np.concatenate((full_obs['observation'], full_obs['desired_goal']), axis=0) 

        step_count = 0
        for t in range(nsteps):
            current_sim_data = self.env.get_sim_data()
            tra_tasks_data.append([current_sim_data, full_obs['desired_goal']])
            
            if use_deterministic:
                actions, values, neglogpacs = self.model.step_max([obs], self.dones)
            else:
                actions, values, neglogpacs = self.model.step([obs], self.dones)

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

            step_count += 1

            rewards = np.array([r, 0.0])

            if r == 0:
                # rewards[-1] = np.sqrt(np.sum(np.full(len(obs), 3)*np.full(len(obs), 3)))   
                fixed_state_f += 3
             
            mb_fixed_state_f.append(fixed_state_f)

            task = [current_sim_data, full_obs['desired_goal']]
            sas = [mb_obs[-1], mb_actions[-1], mb_obs_next[-1]]
            tra_tasks.append(task) 
            tra_tasks_value.append(rewards[-1])
            tra_tasks_sas.append(sas)
            tra_rewards.append(r)

            # tra_r = []
            for m in range(len(tra_tasks_sas)-1, -1, -1):
                start_obs = tra_tasks_sas[m][0]
            #     in_r = 0
            #     if need_pm:
            #         in_r = self.compute_intrinsic_reward(start_obs, actions[0], obs)
            #     tra_r.append(in_r)
                mb_pm_state.append([start_obs, obs])

            #     if len(tra_tasks_sas) - m > 20:
            #         break
            # tra_r = np.asarray(tra_r)
            # rewards[-1] = tra_r.mean()
            if need_pm:
                rewards[-1] = self.compute_intrinsic_reward(mb_obs[-1], mb_actions[-1], mb_obs_next[-1])


            # ##########################################################
            ## for roboschool
            if render:
                self.env.render('human')
                # print(rewards, values)
            
            if self.dones or full_obs['achieved_goal'][-1] < 0.4 or step_count > 50 or t == nsteps-1:
                self.dones = True
                updated = False
                tra_tasks_value = self.update_task_value(tra_tasks, tra_tasks_sas)

                # extra_add = 0
                # for m in range(len(tra_tasks_sas)):
                #     start_obs = tra_tasks_sas[m][0]
                #     tra_r = []
                #     for n in range(m, len(tra_tasks_sas), 1):
                #         _, a, obs_ = tra_tasks_sas[n]
                #         emp_r = self.compute_intrinsic_reward(start_obs, a, obs_)
                #         tra_r.append(emp_r)
                #         mb_pm_state.append([start_obs, obs_])
                #         if m - n > 30:
                #             break
                #     length = len(tra_tasks_sas)-m

                #     state_emp = (np.asarray(tra_r).sum() + extra_add)/length #np.asarray(tra_r).sum()/length
                #     extra_add = np.asarray(tra_r[1:]).sum()
                #     mb_rewards.append([tra_rewards[m], state_emp])
                #     if render:
                #         print(state_emp)  

                # print(tra_rewards)

                # tra_argmax = np.argmax(np.asarray(tra_tasks_value))
                # print(tra_tasks_value[tra_argmax] , self.tasks_value[min_index])
                # if tra_tasks_value[tra_argmax] > self.tasks_value[min_index]:
                #     self.tasks_tra[min_index] = tra_tasks_sas[:tra_argmax]
                #     self.tasks_init_data[min_index] = tra_tasks_data[0]
                #     self.tasks[min_index] = task
                #     self.tasks_value[min_index] = tra_tasks_value[tra_argmax]
                #     self.tasks_sas[min_index] = sas                    
                #     updated = True

                v_before = self.tasks_value[min_index]

                for i in range(len(tra_tasks)):
                    task = tra_tasks[i]
                    value = tra_tasks_value[i]
                    sas = tra_tasks_sas[i]

                    # min_index = np.argmin(np.asarray(self.tasks_value))
                    if value > self.tasks_value[min_index] and r != 0:
                        # print(value, self.tasks_value[min_index], value > self.tasks_value[min_index])
                        self.tasks[min_index] = task
                        self.tasks_value[min_index] = value
                        self.tasks_sas[min_index] = sas
                        updated = True
                if render:
                    if updated:
                        print('    updated', min_index, v_before, self.tasks_value[min_index])
                    else:
                        print('    maxvalu', np.asarray(tra_tasks_value).max())
                step_count = 0
                v_preds = self.model.value([obs])[0]
                # rewards[0] += self.gamma*v_preds[0]
                mb_boost[-1][0] = self.gamma*v_preds[0]

                # if self.dones or step_count > 100:
                    # rewards[-1] += self.gamma*v_preds[-1]
                mb_boost[-1][-1] = self.gamma*v_preds[-1]

                full_obs = self.env.reset()   
                task_prob = np.asarray(self.tasks_value)
                max_prob = task_prob.max()
                mean_prob = task_prob.sum()/np.count_nonzero(self.tasks_value)
                # task_prob = np.clip(task_prob, mean_prob, 100000)
                task_prob = task_prob - task_prob.min()
                task_prob = task_prob/task_prob.sum()        
                # print(task_prob)         
                if np.random.rand() > 0:
                    task_index = np.random.choice(len(self.tasks), 1, p=task_prob)[0]
                    # task_index = np.random.choice(len(self.tasks), 1)[0]
                else:
                    task_index = np.argmax(self.tasks_value)

                # if self.tasks_value[task_index] != -1 and np.count_nonzero(self.tasks_value) > 50:
                if np.random.rand() > random_prob and np.count_nonzero(self.tasks_value) > 3:
                    min_index = task_index
                    task = self.tasks[task_index]
                    if render:
                        print('---', comm_rank, task_index, self.tasks_value[task_index], max_prob, len(self.tasks))
                    # task = self.tasks[task_index]    
                    full_obs = self.env.set_sim_data(task[0], task[1])
                    # full_obs = self.reset_env_sas(self.tasks_init_data[task_index], self.tasks_tra[task_index])
                else:
                    min_index = np.argmin(np.asarray(self.tasks_value))
                    if render:
                        print('---random init')
                obs = np.concatenate((full_obs['observation'], full_obs['desired_goal']), axis=0) 
                self.full_obs = full_obs.copy()

                tra_tasks = []
                tra_tasks_value = []
                tra_tasks_sas = []
                tra_tasks_data = []
                tra_rewards = []

            mb_rewards.append(rewards)

        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_obs_next = np.asarray(mb_obs_next, dtype=self.obs.dtype)
        mb_fixed_state_f = np.asarray(mb_fixed_state_f, dtype=self.obs.dtype)
        mb_pm_state = np.asarray(mb_pm_state, dtype=self.obs.dtype)

        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_boost = np.asarray(mb_boost, dtype=np.float32)

        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value([obs], self.dones)[0]

        #discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0

        for i in range(0, len(mb_obs), 10):
            self.history_states[self.history_states_index] = mb_obs[i]
            self.history_states_index += 1
            self.history_states_index = self.history_states_index % self.history_buff_length


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
            for i in range(len(mb_values)):
                print(mb_rewards[i], mb_returns[i], mb_advs[i], mb_values[i])

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
        for i in range(REWARD_NUM+1):
            model.write_summary('return/mean_ret_'+str(i+1), np.mean(returns[:,i]), step_label)
            model.write_summary('reward/mean_rew_'+str(i+1), np.mean(rewards[:,i]), step_label)
            # model.write_summary('sum_rewards/rewards_'+str(i+1), safemean([epinfo['r'][i] for epinfo in epinfobuf]), step_label)
            # model.write_summary('mean_rewards/rewards_'+str(i+1), safemean([epinfo['r'][i]/epinfo['l'] for epinfo in epinfobuf]), step_label)\
            # model.log_histogram('return_dist/return_'+str(i+1), returns[:,i], step_label)
            # model.log_histogram('adv_dist/adv_'+str(i+1), advs_ori[:,i], step_label)
            # model.log_histogram('values_dist/adv_'+str(i+1), values[:,i], step_label)
            
            # model.write_summary('mean_advs/adv_'+str(i+1), np.mean(advs_ori[:,i]), step_label)
            # model.write_summary('adv_std/adv_'+str(i+1), advs_ori[:,i].std(), step_label)
            
        model.write_summary('sum_rewards/score', safemean([epinfo['r'][-1] for epinfo in epinfobuf]), step_label)
        model.write_summary('sum_rewards/mean_length', safemean([epinfo['l'] for epinfo in epinfobuf]), step_label)

        for (lossval, lossname) in zip(lossvals, model.loss_names):
            model.write_summary('loss/'+lossname, lossval, step_label)


def nimibatch_update(   nbatch, noptepochs, nbatch_train,
                        obs, obs_next, returns, actions, values, advs, rewards, fixed_state_f, neglogpacs,
                        lrnow, cliprangenow, model, update_type = 'all', allow_early_stop = True):
    mblossvals = []
    nbatch_train = int(nbatch_train)

    rewards_mean = rewards.mean()
    # inds = np.arange(nbatch)
    inds = np.arange(len(obs))
    early_stop = False
    for ep in range(noptepochs):
        # print('ep', ep, len(obs), nbatch, nbatch_train)
        # if update_type == 'pm':
        #     print(rewards[:,-1])
        #     rewards[:,-1] = model.pred_error(obs, actions, obs_next)
        #     print(rewards[:,-1])
        np.random.shuffle(inds)
        for start in range(0, len(obs), nbatch_train):
            end = start + nbatch_train
            mbinds = inds[start:end]
            slices = (arr[mbinds] for arr in (obs, obs_next, returns, actions, values, advs, rewards, fixed_state_f, neglogpacs))

            losses = model.train(lrnow, cliprangenow, *slices, train_type=update_type)
            if losses != []:
                mblossvals.append(losses)

    lossvals = np.mean(mblossvals, axis=0)
    return lossvals

def nimibatch_update_pm(noptepochs, nbatch_train, pm_states, lrnow, model, update):
    mblossvals = []
    nbatch_train = int(nbatch_train)

    obs = pm_states[:, 0]
    obs_ = pm_states[:, 1]

    inds = np.arange(len(obs))
    for ep in range(noptepochs):
        np.random.shuffle(inds)
        for start in range(0, len(obs), nbatch_train):
            end = start + nbatch_train
            mbinds = inds[start:end]
            slices = (arr[mbinds] for arr in (obs, obs_))

            losses = model.train_pm(lrnow, *slices)
            if losses != []:
                mblossvals.append(losses)

    lossvals = np.mean(mblossvals, axis=0)
    model.write_summary('loss/forward', lossvals[0], update)

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
