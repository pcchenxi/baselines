import time
import joblib
import numpy as np
import tensorflow as tf

REWARD_NUM = 5
MIN_RETURN = np.array([-1, 0, -200, -100, -1])

class Model(object):
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, max_grad_norm, model_name, lr, need_summary = False):
        
        reward_size = [None, REWARD_NUM]
        # reward_size = [None]
        if need_summary:
            use_gpu = 1
        else:
            use_gpu = 0
        config = tf.ConfigProto(
            inter_op_parallelism_threads=1,
            intra_op_parallelism_threads=1,
            log_device_placement=False,
            device_count = { "GPU": use_gpu } )
        sess = tf.Session(config=config)

        # sess = tf.get_default_session(config=config)

        act_model = policy(sess, ob_space, ac_space, REWARD_NUM, reuse=False, model_name = model_name)
        train_model = policy(sess, ob_space, ac_space, REWARD_NUM, reuse=True, model_name = model_name)

        A = train_model.pdtype.sample_placeholder([None])
        ADV = tf.placeholder(tf.float32, [None], name='ph_adv')
        RETRUN = tf.placeholder(tf.float32, reward_size, name='ph_R')
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None], name='ph_old_neglogpac')
        OLDVPRED = tf.placeholder(tf.float32, reward_size, name='ph_oldvpred')
        LR = tf.placeholder(tf.float32, [], name='ph_lr')
        CLIPRANGE = tf.placeholder(tf.float32, [], name='ph_cliprange')

        neglogpac = train_model.pd.neglogp(A)
        entropy = tf.reduce_mean(train_model.pd.entropy())

        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        vf_losses1 = tf.square(vpred - RETRUN)
        vf_losses2 = tf.square(vpredclipped - RETRUN)
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

        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

        with tf.variable_scope(model_name):
            params = tf.trainable_variables(scope=model_name)
            params_actor = tf.trainable_variables(scope=model_name+'/actor')
            params_value = tf.trainable_variables(scope=model_name+'/value')

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


        # grads_a = tf.gradients(loss_a, params_actor)
        # grads_v = tf.gradients(loss_v, params_value)
        # max_grad_norm = 10
        # if max_grad_norm is not None:
        #     grads_a, _grad_norm = tf.clip_by_global_norm(grads_a, max_grad_norm)
        #     grads_v, _grad_norm = tf.clip_by_global_norm(grads_v, max_grad_norm)
        # grads_a = list(zip(grads_a, params_actor))
        # grads_v = list(zip(grads_v, params_value))
        # optimizer_a = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        # train_a = optimizer_a.apply_gradients(grads_a)

        optimizer_a = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        optimizer_v = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        train_a = optimizer_a.minimize(loss_a, var_list=params_actor)
        train_v = optimizer_v.minimize(loss_v, var_list=params_value)

        def train(lr, cliprange, obs, returns, masks, actions, values, advs, neglogpacs, states=None, train_type='all'):
            # advs_suggest = advs_suggest / (advs_suggest.std() + 1e-8)

            if train_type == 'all':
                advs = (advs - advs.mean()) / (advs.std() + 1e-8)
                td_map = {train_model.X:obs, A:actions, ADV:advs, RETRUN:returns, LR:lr,
                        CLIPRANGE:cliprange, OLDNEGLOGPAC:neglogpacs, OLDVPRED:values}
                if states is not None:
                    td_map[train_model.S] = states
                    td_map[train_model.M] = masks

                return sess.run(
                    [pg_loss, vf_loss, entropy, approxkl, clipfrac, train_a, train_v],
                    td_map                
                )[:-2]
            elif train_type == 'actor':
                advs = (advs - advs.mean()) / (advs.std() + 1e-8)

                td_map = {train_model.X:obs, A:actions, ADV:advs, RETRUN:returns, LR:lr,
                        CLIPRANGE:cliprange, OLDNEGLOGPAC:neglogpacs, OLDVPRED:values}
                if states is not None:
                    td_map[train_model.S] = states
                    td_map[train_model.M] = masks
                # ratio_ = sess.run(ratio, td_map)
                # print('ratio', ratio_)
                return sess.run(
                    [pg_loss, vf_loss, entropy, approxkl, clipfrac, train_a],
                    td_map                
                )[:-1]
            elif train_type == 'value':
                td_map = {train_model.X:obs, RETRUN:returns, LR:lr,
                        CLIPRANGE:cliprange, OLDVPRED:values}
                if states is not None:
                    td_map[train_model.S] = states
                    td_map[train_model.M] = masks

                sess.run(train_v, td_map )
            else:
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! wrong train_type !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')


        def get_values(obs):
            return sess.run(vpred, {train_model.X:obs})

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

            return np.asarray(net_params)

        def replace_params(loaded_params, params_type = 'all'):
            if params_type == 'all':
                sess.run(restores_all, feed_dict={i: d for i, d in zip(network_params_all, loaded_params)})
            elif params_type == 'actor':
                sess.run(restores_actor, feed_dict={i: d for i, d in zip(network_params_actor, loaded_params)})
            elif params_type == 'value':
                sess.run(restores_value, feed_dict={i: d for i, d in zip(network_params_value, loaded_params)})

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
        self.get_neglogpac = get_neglogpac
        self.get_actions_dist = get_actions_dist

        self.initial_state = act_model.initial_state
        self.save = save
        self.load = load
        self.replace_params = replace_params
        self.get_current_params = get_current_params

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

    def run(self, nsteps, is_test = False, use_deterministic = False):

        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_infos = []
        mb_states = self.states
        epinfos = []

        t_s = time.time()
        ret = False
        # self.obs[:] = self.env.reset()
        for t in range(nsteps):
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
                        # rewards[i][2:] += self.gamma[2:]*MIN_RETURN[2:]
                    # rewards[i] += self.gamma*v_preds[i] 

            # if t%100 == 0:
            #     print(t/self.nsteps, time.time() - t_s)
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

        # print(1, time.time() - t_s, len(mb_rewards))
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)

        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, self.states, self.dones)

        #discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        mb_advs_pf = np.zeros_like(mb_rewards)
        lastgaelam = 0

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

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

def display_updated_result( mblossvals, update, log_interval, nsteps, nbatch, 
                            rewards, returns, advs_ori, epinfobuf, model, logger):
    lossvals = np.mean(mblossvals, axis=0)
    if update % log_interval == 0 or update == 1:
        # # ev = explained_variance(values, returns)
        # logger.logkv("serial_timesteps", update*nsteps)
        # logger.logkv("nupdates", update)
        # logger.logkv("total_timesteps", update*nbatch)
        # # logger.logkv("explained_variance", float(ev))
        # # logger.logkv("return_mean", np.mean(returns))
        # logger.logkv('rew_mean', np.mean(rewards))
        # logger.logkv('eprewmean', safemean([epinfo['r'][-1] for epinfo in epinfobuf]))
        # logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
        # for (lossval, lossname) in zip(lossvals, model.loss_names):
        #     logger.logkv(lossname, lossval)
        # logger.dumpkvs()
        
        # step_label = update*nbatch
        step_label = update
        for i in range(REWARD_NUM):
            model.write_summary('return/mean_ret_'+str(i+1), np.mean(returns[:,i]), step_label)
            model.write_summary('sum_rewards/rewards_'+str(i+1), safemean([epinfo['r'][i] for epinfo in epinfobuf]), step_label)
            model.write_summary('mean_rewards/rewards_'+str(i+1), safemean([epinfo['r'][i]/epinfo['l'] for epinfo in epinfobuf]), step_label)\
            # model.log_histogram('return_dist/return_'+str(i+1), returns[:,i], step_label)
            # model.log_histogram('adv_dist/adv_'+str(i+1), advs_ori[:,i], step_label)
            
            # model.write_summary('mean_advs/adv_'+str(i+1), np.mean(advs_ori[:,i]), step_label)
            # model.write_summary('adv_std/adv_'+str(i+1), advs_ori[:,i].std(), step_label)
            
        model.write_summary('sum_rewards/score', safemean([epinfo['r'][-1] for epinfo in epinfobuf]), step_label)
        model.write_summary('sum_rewards/mean_length', safemean([epinfo['l'] for epinfo in epinfobuf]), step_label)

        for (lossval, lossname) in zip(lossvals, model.loss_names):
            model.write_summary('loss/'+lossname, lossval, step_label)


def nimibatch_update(   nbatch, noptepochs, nbatch_train,
                        obs, returns, masks, actions, values, advs, neglogpacs,
                        lrnow, cliprangenow, states, nsteps, model, update_type = 'all'):
    mblossvals = []
    if states is None: # nonrecurrent version
        # inds = np.arange(nbatch)
        inds = np.arange(len(obs))
        for ep in range(noptepochs):
            # print('ep', ep)
            np.random.shuffle(inds)
            for start in range(0, nbatch, nbatch_train):
                end = start + nbatch_train
                mbinds = inds[start:end]
                slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, advs, neglogpacs))

                losses = model.train(lrnow, cliprangenow, *slices, train_type=update_type)
                mblossvals.append(losses)
                    
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
            
    return mblossvals



def compute_advs( advs_ori, dir_w):

    dir_w_length = np.sqrt(np.sum(dir_w*dir_w))
    advs = np.zeros(len(advs_ori))
    # advs_nrom = np.zeros_like(advs_ori)
    # advs_nrom = (advs_ori - np.mean(advs_ori, axis=0)) / (np.std(advs_ori, axis=0) + 1e-8)
    # advs_nrom = advs_ori / (np.std(advs_ori, axis=0) + 1e-8)
    # advs_max = np.abs(np.max(advs_ori, axis=0))

    for t in range(len(advs_ori)):
        advs[t] = np.sum(np.multiply(advs_ori[t], dir_w))/(dir_w_length + 1e-8) # use 11111

        # print (np.array_str(advs_ori[t], precision=3, suppress_small=True) \
        # , np.array_str(advs_nrom[t], precision=3, suppress_small=True) \
        # , advs[t]
        # )

    # advs = (advs - advs.mean()) / (advs.std() + 1e-8)
    # advs = advs / (advs.std() + 1e-8)

    # for t in range(len(advs_ori)):
    #     print (np.array_str(advs_ori[t], precision=3, suppress_small=True) \
    #     , np.array_str(advs_nrom[t], precision=3, suppress_small=True) \
    #     , advs[t]
    #     )

    # print(dir_w)

    return advs

def compute_advs_singleobj(values, returns):
    returns_single = np.sum(returns, axis = 1)
    values_single = np.sum(values, axis = 1)

    advs = returns_single - values_single

    return advs