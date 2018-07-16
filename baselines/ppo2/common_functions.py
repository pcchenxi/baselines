import time
import joblib
import numpy as np
import tensorflow as tf

REWARD_NUM = 5
class Model(object):
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, max_grad_norm, model_name, lr, need_summary = False):
        
        reward_size = [None, REWARD_NUM]
        # reward_size = [None]
        sess = tf.get_default_session()

        act_model = policy(sess, ob_space, ac_space, REWARD_NUM, reuse=False, model_name = model_name)
        train_model = policy(sess, ob_space, ac_space, REWARD_NUM, reuse=True, model_name = model_name)

        A = train_model.pdtype.sample_placeholder([None])
        ADV = tf.placeholder(tf.float32, [None])
        R = tf.placeholder(tf.float32, reward_size)
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        OLDVPRED = tf.placeholder(tf.float32, reward_size)
        LR = tf.placeholder(tf.float32, [])
        CLIPRANGE = tf.placeholder(tf.float32, [])

        neglogpac = train_model.pd.neglogp(A)
        entropy = tf.reduce_mean(train_model.pd.entropy())

        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        vf_losses1 = tf.square(vpred - R)
        vf_losses2 = tf.square(vpredclipped - R)
        # vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        vf_loss = tf.reduce_mean(vf_losses1)

        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        pg_losses = -ADV * ratio
        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))

        loss_a = pg_loss - entropy * ent_coef
        loss_v = vf_loss

        with tf.variable_scope(model_name):
            params = tf.trainable_variables(scope=model_name)
            params_actor = tf.trainable_variables(scope=model_name+'/actor')
            params_value = tf.trainable_variables(scope=model_name+'/value')

        optimizer_a = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        optimizer_v = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        train_a = optimizer_a.minimize(loss_a, var_list=params_actor)
        train_v = optimizer_v.minimize(loss_v, var_list=params_value)

        # loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
        # grads = tf.gradients(loss, params)
        # print(params, len(params))
        # if max_grad_norm is not None:
        #     grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        # grads = list(zip(grads, params))
        # trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        # _train = trainer.apply_gradients(grads)

        def train(lr, cliprange, obs, returns, masks, actions, values, advs, neglogpacs, actions_suggest_list=[], advs_suggest_list=[], states=None):
            # advs = np.clip(advs, -advs.max(), advs.max())
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            # advs_suggest = advs_suggest / (advs_suggest.std() + 1e-8)
            # print('adv min max', advs.min(), advs.max())

            # for actions_suggest, advs_suggest in zip(actions_suggest_list, advs_suggest_list):
            #     positive_adv = []
            #     for i in range(len(advs_suggest)):
            #         if advs_suggest[i] > 0:
            #             positive_adv.append(i)
                
                # if positive_adv != []:
                #     advs_suggest_select = np.take(advs_suggest, positive_adv, axis=0)
                #     advs_suggest_select = advs_suggest_select / (advs_suggest_select.std() + 1e-8)
                #     advs_suggest_select = np.clip(advs_suggest_select, -10000, advs.max())
                #     print('adv_suggest min max', advs_suggest_select.min(), advs_suggest_select.max(), len(advs_suggest_select))

                #     # print(obs.shape, np.take(obs, positive_adv, axis=0).shape)
                #     obs = np.concatenate((obs, np.take(obs, positive_adv, axis=0)), axis=0)
                #     actions = np.concatenate((actions, np.take(actions_suggest, positive_adv, axis=0)), axis=0)
                #     advs = np.concatenate((advs, advs_suggest_select), axis=0)
                #     returns = np.concatenate((returns, np.take(returns, positive_adv, axis=0)), axis=0)
                #     neglogpacs = np.concatenate((neglogpacs, np.take(neglogpacs, positive_adv, axis=0)), axis=0)
                #     values = np.concatenate((values, np.take(values, positive_adv, axis=0)), axis=0)

            # td_map_suggest = {train_model.X:obs, A:actions_suggest, ADV:advs_suggest, LR:lr,
            #         CLIPRANGE:cliprange, OLDNEGLOGPAC:neglogpacs}
            # sess.run(train_a, td_map_suggest)

            # advs = (advs - advs.mean()) / (advs.std() + 1e-8)

            td_map = {train_model.X:obs, A:actions, ADV:advs, R:returns, LR:lr,
                    CLIPRANGE:cliprange, OLDNEGLOGPAC:neglogpacs, OLDVPRED:values}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks

            return sess.run(
                [pg_loss, vf_loss, entropy, approxkl, clipfrac, train_a, train_v],
                td_map                
            )[:-2]
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

        def get_values(obs):
            return sess.run(vpred, {train_model.X:obs})

        def get_actions_dist(obs):
            return sess.run([train_model.sample, train_model.std], {train_model.X:obs})

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
                if load_value == False and find_vf != -1:
                    continue
                
                # find_pi = p.name.find('pi')
                # if find_pi != -1:
                #     print(np.asarray(loaded_p).shape)
                #     noise = np.random.normal(0, 0.005, np.asarray(loaded_p).shape)
                # # if p.name.find('logstd') != -1:
                # #     loaded_p = loaded_p + 0.2
                #     print(p.name, find_vf)
                #     loaded_p += noise
                restores.append(p.assign(loaded_p))
            sess.run(restores)

            # saver.restore(sess, load_path + '.cptk')
            print('loaded')
            # If you want to load weights, also save/load observation scaling inside VecNormalize

        def load_value(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                find_vf = p.name.find('vf')
                if find_vf == -1:
                    continue
                print(p.name, find_vf)
                restores.append(p.assign(loaded_p))
            sess.run(restores)

            # saver.restore(sess, load_path + '.cptk')
            print('loaded')
            # If you want to load weights, also save/load observation scaling inside VecNormalize

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
        self.act_model = act_model
        self.step = act_model.step
        self.step_max = act_model.step_max
        self.value = act_model.value
        self.get_values = get_values
        self.get_actions_dist = get_actions_dist

        self.initial_state = act_model.initial_state
        self.save = save
        self.load = load
        self.load_value = load_value

        if need_summary:
            # self.summary_writer = tf.summary.FileWriter('../model/log', sess.graph)   
            self.summary_writer = tf.summary.FileWriter('../model/log')   
            self.write_summary = write_summary
            self.log_histogram = log_histogram
        
        saver = tf.train.Saver()

        tf.global_variables_initializer().run(session=sess) #pylint: disable=E1101


class Runner(object):

    def __init__(self, *, env, model, nsteps, gamma, lam):
        self.env = env
        self.model = model
        nenv = env.num_envs
        self.nenv = nenv
        self.obs = np.zeros((nenv,) + env.observation_space.shape, dtype=model.train_model.X.dtype.name)
        self.obs[:] = env.reset()
        self.gamma = np.array(gamma)
        self.lam = lam
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

    def run(self, is_test = False, use_deterministic = False):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_infos = []
        mb_states = self.states
        epinfos = []

        t_s = time.time()
        ret = False
        # self.obs[:] = self.env.reset()
        for t in range(self.nsteps):
            if use_deterministic:
                actions, values, self.states, neglogpacs = self.model.step_max(self.obs, self.states, self.dones)
            else:
                actions, values, self.states, neglogpacs = self.model.step(self.obs, self.states, self.dones)

            # print(values)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)

            self.obs[:], rewards, self.dones, infos = self.env.step(actions)

            rewards = rewards[:,:-1]
            ##########################################################
            ### for roboschool
            for i in range(len(self.dones)):
                if self.dones[i]:
                    # print(mins[:,0], maxs[:,0], values[:,0], rewards[:,0])
                    # print(rewards[0], values[0])
                    v_preds = []
                    if v_preds == []:
                        v_preds = self.model.value(self.obs)
                    if rewards[i][0] > 0:  
                        rewards[i] += self.gamma*v_preds[i] 
                    else:
                        rewards[i][2:] += self.gamma[2:]*v_preds[i][2:]
                    # rewards[i] += self.gamma*v_preds[i] 

            if t%100 == 0:
                print(t/self.nsteps, time.time() - t_s)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
            mb_infos.append(infos)        

            if is_test:
                for i in range(len(self.dones)):
                    if self.dones[i]:
                        ret = True            
                if ret:
                    print(rewards)
                    print('score', infos[-1]['episode']['r'])
                    break

        print(1, time.time() - t_s, len(mb_rewards))
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)

        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, self.states, self.dones)

        # print('last', last_values)
        #discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        mb_advs_pf = np.zeros_like(mb_rewards)
        lastgaelam = 0

        # print('in runner',np.asarray(mb_rewards).shape)
        # print('         ',np.asarray(mb_values).shape)

        for t in reversed(range(len(mb_values))):
            nextnonterminal = np.zeros(last_values.shape)
            if t == len(mb_values) - 1:
                nextnonterminal[:,0] = nextnonterminal[:,1] = nextnonterminal[:,2] = nextnonterminal[:,3] = nextnonterminal[:,4] = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal[:,0] = nextnonterminal[:,1] = nextnonterminal[:,2] = nextnonterminal[:,3] = nextnonterminal[:,4] = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]

            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]

            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values

        # if is_test:
        #     for t in range(len(mb_returns)):
        #         print(''
        #         # , advs[t]
        #         # , v_score[t]
        #         # , ret_score[t]
        #         # , v_score_pf[t]
        #         # , ret_score_final[t]
        #         , np.array_str(mb_returns[t], precision=3, suppress_small=True) \
        #         # # , np.array_str(ratios, precision=3, suppress_small=True) \
        #         # # , np.array_str(ratios_mean, precision=3, suppress_small=True) \
        #         , np.array_str(advs[t], precision=3, suppress_small=True) \
        #         )

        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_rewards, mb_neglogpacs)),
            mb_states, epinfos, ret)

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

def compute_advs(max_return, values, advs_ori, action_diff, action_std):
    # towards to max point
    dir_v = np.asarray(max_return - values)
    dir_v_length = np.sqrt(np.sum(np.multiply(dir_v, dir_v), axis=1)) 

    advs_dir_norm = np.zeros_like(advs_ori)
    for t in range(len(values)):
        advs_dir_norm[t] = dir_v[t]/(dir_v_length[t] + 1e-8)
    mean_dir = np.mean(advs_dir_norm, axis=0)
    mean_dir_length = np.sqrt(np.sum(mean_dir*mean_dir))
    print('norm dir', np.array_str(mean_dir, precision=3, suppress_small=True))

    # towards to max evaluation gradient
    # grad_v = values * 1
    grad_v = values * 1.1   

    # for i in range(REWARD_NUM):
    #     grad_v[:, i] = np.clip(grad_v[:, i], -10000, max_return[i])
    grad_v = grad_v - values
    grad_v_length = np.sqrt(np.sum(np.multiply(grad_v, grad_v), axis=1)) 

    g_v = np.array([1, 1, 1, 1, 1])
    g_v_length = np.sqrt(np.sum(g_v*g_v))
    # advs_grad = np.sum(advs_ori, axis=1)/math.sqrt(REWARD_NUM)

    advs_dir = np.zeros(len(values))
    advs_grad = np.zeros(len(values))
    advs = np.zeros(len(values))
    # advs_nrom = (advs_ori - np.mean(advs_ori, axis=0)) / (np.std(advs_ori, axis=0) + 1e-8)

    for t in range(len(values)):
        # advs_grad[t] = np.sum(np.multiply(advs_ori[t], grad_v[t]))/(grad_v_length[t] + 1e-8)
        # advs_grad[t] = np.sum(np.multiply(advs_ori[t], g_v))/(g_v_length + 1e-8) # use 11111
        advs_grad[t] = np.sum(np.multiply(advs_ori[t], mean_dir_length))/(mean_dir_length + 1e-8) # use max point
        advs_dir[t] = np.sum(np.multiply(advs_ori[t], dir_v[t]))/(dir_v_length[t] + 1e-8)

        advs[t] = 0.0 * advs_dir[t] + 1.0 * advs_grad[t]
        if action_diff != []:
            if np.all(action_diff[t] < action_std[t] * 0.6):
                advs[t] = -1
                # print(action_diff[t], action_std[t], advs[t])

    return advs