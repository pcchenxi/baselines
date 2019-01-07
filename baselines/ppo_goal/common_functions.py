import joblib
import time, os
import os.path as osp
import numpy as np
import tensorflow as tf

from mpi4py import MPI

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()

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

        neglogpac = train_model.pd.neglogp(train_model.A)
        entropy = tf.reduce_mean(train_model.pd.entropy())

        vpred = train_model.vf
        # vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        vf_losses1 = tf.square(vpred - RETRUN)
        # vf_losses2 = tf.square(vpredclipped - RETRUN)
        # vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        vf_loss = tf.reduce_mean(vf_losses1)

        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        pg_losses = -ADV * ratio
        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        # pg_loss = tf.reduce_mean(pg_losses2)
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))

        loss_a = pg_loss - entropy * ent_coef
        loss_v = vf_loss

        loss_forward = tf.reduce_mean(tf.square(train_model.x_next_feature_pred - train_model.x_next_feature))

        self.loss_names = ['policy_loss', 'value_loss', 'forward_loss', 'policy_entropy', 'approxkl', 'clipfrac', 'grad_norm']

        with tf.variable_scope(model_name):
            params = tf.global_variables(scope=model_name)
            params_actor = tf.global_variables(scope=model_name+'/actor')
            params_value = tf.global_variables(scope=model_name+'/value')
            params_forward = tf.global_variables(scope=model_name+'/forward_dynamic')


        grads_a = tf.gradients(loss_a, params_actor)
        max_grad_norm = 1
        if max_grad_norm is not None:
            grads_a, _grad_norm_a = tf.clip_by_global_norm(grads_a, max_grad_norm)
        grads_a = list(zip(grads_a, params_actor))


        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        print(update_ops)

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


        def train(lr, cliprange, obs, obs_next, returns, actions, values, advs, neglogpacs, train_type='all'):
            # advs_suggest = advs_suggest / (advs_suggest.std() + 1e-8)
            # obs_ret = np.concatenate((obs, returns[:, 0:-1]), axis=1)
            obs_ret = np.concatenate((obs, actions), axis=1)

            if train_type == 'all':
                advs = (advs - advs.mean()) / (advs.std() + 1e-8)                
                td_map = {train_model.is_training: True, train_model.X:obs, train_model.X_NEXT:obs_next, train_model.A:actions, ADV:advs, RETRUN:returns, 
                        LR:lr, CLIPRANGE:cliprange, OLDNEGLOGPAC:neglogpacs, OLDVPRED:values}
                # vp, ret, v_l1 = sess.run(
                #     [OLDNEGLOGPAC, neglogpac, ratio],
                #     td_map                
                # )
                # print(vp.shape, ret.shape, v_l1.shape)

                return sess.run(
                    [pg_loss, loss_v, loss_forward, entropy, approxkl, clipfrac, _grad_norm_a, train_a, train_v, train_forward],
                    td_map                
                )[:-3]
            # if train_type == 'value':
            #     advs = (advs - advs.mean()) / (advs.std() + 1e-8)                
            #     td_map = {train_model.is_training: True, train_model.X:obs, train_model.A:actions, ADV:advs, RETRUN:returns, 
            #             LR:lr, CLIPRANGE:cliprange, OLDNEGLOGPAC:neglogpacs, OLDVPRED:values}

            #     return sess.run(
            #         [pg_loss, loss_v, entropy, approxkl, clipfrac, _grad_norm_a, train_v],
            #         td_map                
            #     )[:-1]

        def get_neglogpac(obs, actions):
            return sess.run(neglogpac, {train_model.X:obs, A:actions})

        def get_actions_dist(obs):
            return sess.run([train_model.mean, train_model.std], {train_model.X:obs})

        def save(save_path, save_path_base):
            # saver.save(sess, save_path + '.cptk') 
            ps = sess.run(params)
            joblib.dump(ps, save_path)
            joblib.dump(ps, save_path_base)

        def load(load_path, load_value = True):
            loaded_params = joblib.load(load_path)
            print('parama', len(loaded_params), load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                find_vf = p.name.find('vf')
                # if find_vf != -1:
                #     print(p.name, loaded_p)
                if load_value == False and find_vf != -1:
                    continue

                restores.append(p.assign(loaded_p))
            sess.run(restores)

            # saver.restore(sess, load_path + '.cptk')
            print('loaded')
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
                sess.run(restores_fm, feed_dict={i: d for i, d in zip(network_params_forward, loaded_params)})

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
        self.train_model = train_model
        # self.act_model = act_model
        self.step = train_model.step
        self.step_max = train_model.step_max
        self.value = train_model.value
        self.get_neglogpac = get_neglogpac
        self.get_actions_dist = get_actions_dist

        self.save = save
        self.load = load
        self.replace_params = replace_params
        self.get_current_params = get_current_params

        self.state_action_pred = train_model.state_action_pred
        self.state_feature = train_model.state_feature

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

    def __init__(self, *, env, model, nsteps, gamma, lam):
        self.env = env
        self.model = model
        nenv = 1 #env.num_envs
        self.nenv = nenv
        self.obs = np.zeros([1, 28], dtype=model.train_model.X.dtype.name)
        # self.obs[:] = env.reset()
        self.gamma = np.full(REWARD_NUM, gamma)
        self.gamma[-1] = 0.98
        self.lam = lam
        self.nsteps = nsteps
        self.dones = [False for _ in range(nenv)]

        self.tasks_sas = []
        self.tasks = []
        self.tasks_vlaue = []
        self.min_value = 0
        self.index = 0
        self.task_length = 20
        self.buffer_full = False

    def init_task_pool(self, nsteps, model_old, render = False):
        self.tasks = []
        self.tasks_vlaue = []

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
            #     full_obs = self.env.set_sim_data(task[0], task[1])

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
                                self.tasks_vlaue.append(real_value)    
                    if render:
                        print('task length', len(tra_tasks), len(self.tasks))
                    break   

            if len(self.tasks) > 300:
                break

        if len(self.tasks) > 0:
            print(len(self.tasks), np.mean(self.tasks_vlaue), np.max(self.tasks_vlaue), np.min(self.tasks_vlaue))      
            self.tasks_vlaue = -np.asarray(self.tasks_vlaue)
            mean_v = self.tasks_vlaue.mean()
            # final_task, final_v = [], []
            # for task, v in zip(self.tasks, self.tasks_vlaue):
            #     if v > mean_v:
            #         final_task.append(task)
            #         final_v.append(v)
            # self.tasks = final_task.copy()
            # self.tasks_vlaue = final_v.copy()
            self.tasks_vlaue = self.tasks_vlaue - self.tasks_vlaue.min()
            self.tasks_vlaue = self.tasks_vlaue/self.tasks_vlaue.sum()            
            print(len(self.tasks), np.mean(self.tasks_vlaue), np.max(self.tasks_vlaue), np.min(self.tasks_vlaue))      


    def run(self, nsteps, is_test = False, use_deterministic = False, render = False):
        mb_obs, mb_obs_next, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[],[]
        mb_infos = []
        epinfos = []

        # self.tasks = []
        # self.tasks_vlaue = []
        
        # update task value
        for i in range(len(self.tasks)):
            if self.tasks_vlaue[i] < 0:
                continue
            s, a, s_ = self.tasks_sas[i]
            next_statef_pred = self.model.state_action_pred([s], [a])
            next_statef = self.model.state_feature([s_])
            diff_f = (next_statef_pred - next_statef)
            diff_f_norm = np.sqrt(np.sum(diff_f*diff_f))
            rewards_norm = diff_f_norm
            self.tasks_vlaue[i] = rewards_norm   

        print('average task value', np.asarray(self.tasks_vlaue).mean())

        full_obs = self.env.reset()  
        if len(self.tasks) > 0:
            task_prob = np.asarray(self.tasks_vlaue)
            task_prob = task_prob - task_prob.min()
            task_prob = task_prob/task_prob.sum() 

            task_index = np.random.choice(len(self.tasks), 1, p=task_prob)[0]
            task = self.tasks[task_index]     
            full_obs = self.env.set_sim_data(task[0], task[1])
            # joblib.dump(self.tasks, '../model/checkpoints/tasks')
            # joblib.dump(self.tasks_sas, '../model/checkpoints/tasks_sas')
            # joblib.dump(self.tasks_vlaue, '../model/checkpoints/tasks_value')
            min_index = np.argmin(np.asarray(self.tasks_vlaue))

        obs = np.concatenate((full_obs['observation'], full_obs['desired_goal']), axis=0) 
        actions, values, neglogpacs = self.model.step([obs], self.dones)

        sas = [obs, actions, obs]
        task_obs = obs
        task_value = -1 #mb_values[-1][-1]  
        current_sim_data = self.env.get_sim_data()
        task = [current_sim_data, full_obs['desired_goal']]        
        for i in range(self.task_length):
            self.tasks.append(task)
            self.tasks_vlaue.append(task_value) 
            self.tasks_sas.append(sas)            

        min_index = np.argmin(np.asarray(self.tasks_vlaue))

        t_s = time.time()
        ret = False
        count_down = 0
        
        for t in range(nsteps):
            # if t%500 == 0:
                # print(t)
            current_sim_data = self.env.get_sim_data()
            
            if use_deterministic:
                actions, values, neglogpacs = self.model.step_max([obs], self.dones)
            else:
                actions, values, neglogpacs = self.model.step([obs], self.dones)

            mb_obs.append(obs.copy())
            mb_actions.append(actions[0])
            mb_values.append(values[0])
            mb_neglogpacs.append(neglogpacs[0])
            mb_dones.append(self.dones)

            next_statef_pred = self.model.state_action_pred([obs], actions)

            full_obs, r, self.dones, infos = self.env.step(actions[0])
            obs = np.concatenate((full_obs['observation'], full_obs['desired_goal']), axis=0) 
            mb_obs_next.append(obs.copy())

            rewards = np.array([r, 0.0])

            object_related_index = [3,4,5,11,12,13,14,15,16,17,18,19]
            next_statef = self.model.state_feature([obs])
            # next_statef_pred = self.model.state_action_pred([obs], actions)

            diff_f = (next_statef_pred - next_statef) #np.take((next_statef_pred - next_statef), object_related_index)
            diff_f_norm = np.sqrt(np.sum(diff_f*diff_f))
            rewards_norm = diff_f_norm
            rewards[-1] = rewards_norm

            sas = [mb_obs[-1], mb_actions[-1], mb_obs_next[-1]]
            task_obs = mb_obs[-1]
            task_value = rewards_norm #mb_values[-1][-1]  
            task = [current_sim_data, full_obs['desired_goal']]

            if rewards_norm > self.tasks_vlaue[min_index]:
                print('replace', min_index, rewards_norm, self.tasks_vlaue[min_index])
                self.tasks[min_index] = task
                self.tasks_vlaue[min_index] = task_value
                self.tasks_sas[min_index] = sas
                # count_down = 5

            # ##########################################################
            ## for roboschool
            if render:
                self.env.render('human')
                # print(rewards)

            v_preds = []
            if self.dones or full_obs['achieved_goal'][-1] < 0.42:
                v_preds = self.model.value([obs])[0]
                rewards[0] += self.gamma*v_preds[0]
                if self.dones:
                    rewards[-1] += self.gamma*v_preds[-1]

                full_obs = self.env.reset()   
                if len(self.tasks) > 0:
                    min_index = np.argmin(np.asarray(self.tasks_vlaue))
                if np.random.rand() > 0.5 and len(self.tasks) > 10:
                    task_prob = np.asarray(self.tasks_vlaue)
                    task_prob = task_prob - task_prob.min()
                    task_prob = task_prob/task_prob.sum()        
                    # print(task_prob)             
                    task_index = np.random.choice(len(self.tasks), 1, p=task_prob)[0]
                    min_index = task_index
                    print(task_index, self.tasks_vlaue[task_index], len(self.tasks))
                    task = self.tasks[task_index]    
                    full_obs = self.env.set_sim_data(task[0], task[1])
                else:
                    print('random init')
                obs = np.concatenate((full_obs['observation'], full_obs['desired_goal']), axis=0) 

            # print(infos, full_obs['desired_goal'], full_obs['achieved_goal'])
            mb_rewards.append(rewards)

        # print(1, time.time() - t_s, len(mb_rewards))
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_obs_next = np.asarray(mb_obs_next, dtype=self.obs.dtype)

        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)

        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value([obs], self.dones)[0]

        #discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0


        for t in reversed(range(len(mb_values))):
            if t == len(mb_values) - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]     
                nextvalues = mb_values[t+1]

            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]

            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam

        mb_returns = mb_advs + mb_values

        # print(mb_values.shape, mb_rewards.shape, mb_returns.shape)
        # for i in range(len(mb_values)):
        #     print(mb_rewards[i], mb_returns[i], mb_dones[i], mb_advs[i])

        return (mb_obs, mb_obs_next, mb_returns, mb_dones, mb_actions, mb_values, mb_advs, mb_rewards, mb_neglogpacs, epinfos, ret)
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
            model.write_summary('sum_rewards/rewards_'+str(i+1), safemean([epinfo['r'][i] for epinfo in epinfobuf]), step_label)
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
                        obs, obs_next, returns, actions, values, advs, neglogpacs,
                        lrnow, cliprangenow, nsteps, model, update_type = 'all', allow_early_stop = True):
    mblossvals = []
    nbatch_train = int(nbatch_train)

    # inds = np.arange(nbatch)
    inds = np.arange(len(obs))
    early_stop = False
    for ep in range(noptepochs):
        # print('ep', ep, len(obs), nbatch, nbatch_train)
        np.random.shuffle(inds)
        for start in range(0, len(obs), nbatch_train):
            params_all_base = model.get_current_params(params_type='all')
            end = start + nbatch_train
            mbinds = inds[start:end]
            slices = (arr[mbinds] for arr in (obs, obs_next, returns, actions, values, advs, neglogpacs))

            losses = model.train(lrnow, cliprangenow, *slices, train_type=update_type)
            mblossvals.append(losses)

    lossvals = np.mean(mblossvals, axis=0)
    return lossvals



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
