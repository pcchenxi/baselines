import numpy as np
import tensorflow as tf
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from baselines.common.distributions import make_pdtype


def nature_cnn(unscaled_images):
    """
    CNN from Nature paper.
    """
    scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
    activ = tf.nn.relu
    h = activ(conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2)))
    h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2)))
    h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2)))
    h3 = conv_to_fc(h3)
    return activ(fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2)))

class LnLstmPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False):
        nenv = nbatch // nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(X)
            xs = batch_to_seq(h, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lnlstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            pi = fc(h5, 'pi', nact)
            vf = fc(h5, 'v', 1)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, v0, snew, neglogp0], {X:ob, S:state, M:mask})

        def value(ob, state, mask):
            return sess.run(v0, {X:ob, S:state, M:mask})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class LstmPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False):
        nenv = nbatch // nsteps

        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(X)
            xs = batch_to_seq(h, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            pi = fc(h5, 'pi', nact)
            vf = fc(h5, 'v', 1)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, v0, snew, neglogp0], {X:ob, S:state, M:mask})

        def value(ob, state, mask):
            return sess.run(v0, {X:ob, S:state, M:mask})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class CnnPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False): #pylint: disable=W0613
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(X)
            pi = fc(h, 'pi', nact, init_scale=0.01)
            vf = fc(h, 'v', 1)[:,0]

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value


activ = tf.nn.relu
class MlpPolicy(object):
    def create_obs_feature_net(self, obs, is_training, reuse=False, use_bn = False):
        with tf.variable_scope('obs_feature', reuse=reuse):    
            h1_obsf = self.dense_layer_bn(obs, 64, 'obs_f_h1', tf.nn.relu, is_training, use_bn=use_bn) #activ(fc(X, 'vf_fc1', nh=128, init_scale=np.sqrt(2)))
            h2_obsf = self.dense_layer_bn(h1_obsf, 32, 'obs_f_h2', tf.nn.relu, is_training, use_bn=use_bn) #activ(fc(h1_val, 'vf_fc2', nh=64, init_scale=np.sqrt(2)))
        return h2_obsf

    def dense_layer(self, input_s, output_size, name, activation):
        out = tf.layers.dense(input_s, output_size, activation, name=name)
        return out

    def dense_layer_bn(self, input_s, output_size, name, activation, training, use_bn = False):
        out = tf.layers.dense(input_s, output_size, name=name)
        if use_bn:
            out = tf.layers.batch_normalization(out, name=name+'_bn', training=training)
        out = activation(out)
        return out

    def __init__(self, sess, ob_space, ac_space, REWARD_NUM, reuse=False, model_name="model"): #pylint: disable=W0613
        # with tf.device('/device:GPU:0'):
        ob_shape = (None,) + ob_space.shape
        ac_shape = (None,) + ac_space.shape
        print('ob_space.shape', ob_space.shape)

        actdim = ac_space.shape[0]
        is_training = tf.placeholder_with_default(False, shape=(), name='training')

        X = tf.placeholder(tf.float32, ob_shape, name='Ob') #obs
        X_NEXT = tf.placeholder(tf.float32, ob_shape, name='Ob_next') #obs
        REWARD = tf.placeholder(tf.float32, [None, REWARD_NUM-1], name='ph_Rew')

        A = tf.placeholder(tf.float32, ac_shape, name='action') #obs   train_model.pdtype.sample_placeholder([None])
        
        with tf.variable_scope(model_name, reuse=reuse):
            with tf.variable_scope('actor', reuse=reuse):
                h1_act = self.dense_layer_bn(X, 128, 'pi_fc1', tf.nn.relu, is_training)  #activ(fc(X, 'pi_fc1', nh=256, init_scale=np.sqrt(2)))
                h2_act = self.dense_layer_bn(h1_act, 64, 'pi_fc2', tf.nn.relu, is_training) #activ(fc(h1_act, 'pi_fc2', nh=128, init_scale=np.sqrt(2)))
                pi = self.dense_layer(h2_act, actdim, 'pi', None) #fc(h2_act, 'pi', actdim, init_scale=0.01)
                logstd = tf.get_variable(name="logstd", shape=[1, actdim],
                    initializer=tf.zeros_initializer())    

            with tf.variable_scope('value', reuse=reuse):        
                h1_val = self.dense_layer_bn(X, 128, 'vf_fc1', tf.nn.relu, is_training) #activ(fc(X, 'vf_fc1', nh=128, init_scale=np.sqrt(2)))
                h2_val = self.dense_layer_bn(h1_val, 64, 'vf_fc2', tf.nn.relu, is_training) #activ(fc(h1_val, 'vf_fc2', nh=64, init_scale=np.sqrt(2)))
                vf_1 = self.dense_layer(h2_val, REWARD_NUM-1, 'vf', None) #fc(h2_val, 'vf', REWARD_NUM-1)
                vf_ri = self.dense_layer(h2_val, 1, 'vf_ri', None) #fc(h2_val, 'vf_ri', 1)

                vf = tf.concat([vf_1, vf_ri], axis=1)      

            with tf.variable_scope('inverse_dynamic', reuse=reuse): 
                x_feature = self.create_obs_feature_net(X, is_training, reuse=False)
                x_next_feature = X_NEXT #self.create_obs_feature_net(X_NEXT, is_training, reuse=True)
                combined_state_f = tf.concat([x_feature, x_next_feature], axis=1)
# 
                # next_state_reward = tf.concat([x_next_feature, REWARD], axis=1)   
                # x_next_feature = self.dense_layer_bn(next_state_reward, 32, 'xnf', tf.nn.relu, is_training)
                # x_next_feature = self.dense_layer(x_next_feature, 32, 'xnf2', None)


                # x_next_feature_reward = tf.concat([x_next_feature, REWARD], axis=1)
                # x_next_feature_reward_fc = self.dense_layer_bn(x_next_feature_reward, 32, 'next_f_r_fc', tf.nn.relu, is_training) #fc(x_next_feature_reward, 'next_f_r_fc', 32)
                # combined_state_f = tf.concat([x_feature, x_next_feature_reward_fc], axis=1)

                rp_h1 = self.dense_layer_bn(combined_state_f, 64, 'rp_h1', tf.nn.relu, is_training) #activ(fc(combined_state_f, 'rp_h1', nh=64, init_scale=np.sqrt(2)))
                rp_h2 = self.dense_layer_bn(rp_h1, 32, 'rp_h2', tf.nn.relu, is_training) #activ(fc(rp_h1, 'rp_h2', nh=32, init_scale=np.sqrt(2)))
                inverse_dynamic_pred = self.dense_layer(rp_h2, actdim, 'inverse_pred', None) #fc(rp_h2, 'inverse_pred', ac_space.shape[0])
                # inverse_dynamic_pred = self.dense_layer(rp_h2, REWARD_NUM-1, 'inverse_pred', None) #fc(rp_h2, 'inverse_pred', REWARD_NUM-1)

            with tf.variable_scope('forward_dynamic', reuse=reuse):   
                state_action = tf.concat([X, A], axis=1)     
                # next_state_reward = tf.concat([x_next_feature, REWARD], axis=1)   

                h1_fd = self.dense_layer_bn(state_action, 64, 'fd_h1', tf.nn.relu, is_training, use_bn=False) #activ(fc(state_action, 'fd_h1', nh=64, init_scale=np.sqrt(2)))
                h2_fd = self.dense_layer_bn(h1_fd, 32, 'fd_h2', tf.nn.relu, is_training, use_bn=False) #activ(fc(h1_fd, 'fd_h2', nh=32, init_scale=np.sqrt(2)))
                # x_next_feature_pred = self.dense_layer(h2_fd, 32, 'next_state_f', None) #fc(h2_fd, 'next_state_f', 32)
                x_next_feature_pred = fc(h2_fd, 'next_state_f', ob_space.shape[0])



        pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pdparam)

        a0 = self.pd.sample()
        a1 = self.pd.mode()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            a = np.clip(a, -2, 2)
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        def get_state_action_prediction(ob, action, *_args, **_kwargs):
            # return sess.run(x_next_feature_pred, {X:ob, REWARD:reward})
            return sess.run(x_next_feature_pred, {X:ob, A:action})

        def get_state_feature(ob, *_args, **_kwargs):
            return sess.run(x_feature, {X:ob})

        def get_next_state_feature(ob, reward, *_args, **_kwargs):
            # return sess.run(x_next_feature_reward_fc, {X_NEXT:ob, REWARD:reward})
            return sess.run(x_next_feature, {X_NEXT:ob, REWARD:reward})

        def get_inverse_dynamic_pred(ob, ob_next, *_args, **_kwargs):
            # return sess.run(x_next_feature_reward_fc, {X_NEXT:ob, REWARD:reward})
            return sess.run(inverse_dynamic_pred, {X:ob, X_NEXT:ob_next})

        def step_max(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a1, vf, neglogp0], {X:ob})
            # a = np.clip(a, -1, 1)
            return a, v, self.initial_state, neglogp

        self.REWARD = REWARD
        self.X = X
        self.X_NEXT = X_NEXT
        self.is_training = is_training
        self.A = A
        self.pi = pi
        self.std = self.pd.std
        self.vf = vf
        self.step = step
        self.step_max = step_max
        self.value = value
        self.mean = a1

        self.inverse_dynamic_pred = inverse_dynamic_pred
        self.x_feature = x_feature
        self.x_next_feature = x_next_feature
        self.x_next_feature_pred = x_next_feature_pred

        self.state_feature = get_state_feature
        self.state_action_pred = get_state_action_prediction
        self.next_state_feature = get_next_state_feature
        self.next_action_pred = get_inverse_dynamic_pred



activ = tf.nn.relu
class MlpPolicy_Goal(object):
    def create_obs_feature_net(self, obs, is_training, reuse=False):
        with tf.variable_scope('obs_feature', reuse=reuse):    
            h1_obsf = self.dense_layer_bn(obs, 64, 'obs_f_h1', tf.nn.relu, is_training) #activ(fc(X, 'vf_fc1', nh=128, init_scale=np.sqrt(2)))
            h2_obsf = self.dense_layer_bn(h1_obsf, 32, 'obs_f_h2', tf.nn.relu, is_training) #activ(fc(h1_val, 'vf_fc2', nh=64, init_scale=np.sqrt(2)))
        return h2_obsf

    def dense_layer(self, input_s, output_size, name, activation):
        out = tf.layers.dense(input_s, output_size, activation, name=name)
        return out

    def dense_layer_bn(self, input_s, output_size, name, activation, training):
        out = tf.layers.dense(input_s, output_size, name=name)
        # out = tf.layers.batch_normalization(out, name=name+'_bn', training=training)
        out = activation(out)
        return out

    def __init__(self, sess, ob_space, ac_space, goal_space, REWARD_NUM, reuse=False, model_name="model"):
        # with tf.device('/device:GPU:0'):
        ob_shape = [None, ob_space.shape[0]+goal_space.shape[0]]
        ac_shape = (None,) + ac_space.shape

        print('ob_space.shape', ob_shape)

        actdim = ac_space.shape[0]
        is_training = tf.placeholder_with_default(False, shape=(), name='training')

        X = tf.placeholder(tf.float32, ob_shape, name='Ob') #obs
        # REWARD = tf.placeholder(tf.float32, [None, REWARD_NUM-1], name='ph_Rew')

        A = tf.placeholder(tf.float32, ac_shape, name='action') #obs   train_model.pdtype.sample_placeholder([None])
        
        with tf.variable_scope(model_name, reuse=reuse):
            with tf.variable_scope('actor', reuse=reuse):
                h1_act = self.dense_layer_bn(X, 128, 'pi_fc1', tf.nn.relu, is_training)  #activ(fc(X, 'pi_fc1', nh=256, init_scale=np.sqrt(2)))
                h2_act = self.dense_layer_bn(h1_act, 64, 'pi_fc2', tf.nn.relu, is_training) #activ(fc(h1_act, 'pi_fc2', nh=128, init_scale=np.sqrt(2)))
                pi = self.dense_layer(h2_act, actdim, 'pi', None) #fc(h2_act, 'pi', actdim, init_scale=0.01)
                logstd = tf.get_variable(name="logstd", shape=[1, actdim],
                    initializer=tf.zeros_initializer())    

            with tf.variable_scope('value', reuse=reuse):        
                h1_val = self.dense_layer_bn(X, 128, 'vf_fc1', tf.nn.relu, is_training) #activ(fc(X, 'vf_fc1', nh=128, init_scale=np.sqrt(2)))
                h2_val = self.dense_layer_bn(h1_val, 64, 'vf_fc2', tf.nn.relu, is_training) #activ(fc(h1_val, 'vf_fc2', nh=64, init_scale=np.sqrt(2)))
                vf_1 = self.dense_layer(h2_val, 1, 'vf', None) #fc(h2_val, 'vf', REWARD_NUM-1)
                # vf_ri = self.dense_layer(h2_val, 1, 'vf_ri', None) #fc(h2_val, 'vf_ri', 1)

                vf = vf_1 #tf.concat([vf_1, vf_ri], axis=1)      

        pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pdparam)

        a0 = self.pd.sample()
        a1 = self.pd.mode()
        neglogp0 = self.pd.neglogp(a0)

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            a = np.clip(a, -2, 2)
            return a, v, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        def step_max(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a1, vf, neglogp0], {X:ob})
            # a = np.clip(a, -1, 1)
            return a, v, neglogp

        self.X = X
        self.is_training = is_training
        self.A = A
        self.pi = pi
        self.std = self.pd.std
        self.vf = vf
        self.step = step
        self.step_max = step_max
        self.value = value
        self.mean = a1



class MlpPolicy_Goal(object):
    def create_obs_feature_net(self, obs, is_training, units=[128, 64], reuse=False, act = tf.nn.relu, use_bn = False):
        with tf.variable_scope('obs_feature', reuse=reuse):    
            for i in range(len(units)):
                if i == 0:
                    hd = self.dense_layer_bn(obs, units[i], act, is_training, use_bn = False) #activ(fc(X, 'vf_fc1', nh=128, init_scale=np.sqrt(2)))
                else:
                    hd = self.dense_layer_bn(hd, units[i], act, is_training, use_bn = False) #activ(fc(X, 'vf_fc1', nh=128, init_scale=np.sqrt(2)))

            # h2_obsf = self.dense_layer_bn(h1_obsf, 64, 'obs_f_h2', tf.nn.relu, is_training) #activ(fc(h1_val, 'vf_fc2', nh=64, init_scale=np.sqrt(2)))
        return hd

    def dense_layer(self, input_s, output_size, activation):
        out = tf.layers.dense(input_s, output_size, activation)
        return out

    def dense_layer_bn(self, input_s, output_size, activation, training, use_bn = False):
        out = tf.layers.dense(input_s, output_size, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32))
        if use_bn:
            out = tf.layers.batch_normalization(out, name=name+'_bn', training=training)
        out = activation(out)
        return out

    def __init__(self, sess, ob_space, ac_space, goal_space, REWARD_NUM, reuse=False, model_name="model"):
        # with tf.device('/device:GPU:0'):
        ob_shape = [None, ob_space.shape[0]+goal_space.shape[0]]
        ac_shape = (None,) + ac_space.shape

        print('ob_space.shape', ob_shape)

        actdim = ac_space.shape[0]
        is_training = tf.placeholder_with_default(False, shape=(), name='training')

        X = tf.placeholder(tf.float32, ob_shape, name='Ob') #obs
        X_NEXT = tf.placeholder(tf.float32, ob_shape, name='Ob_next') #obs

        # REWARD = tf.placeholder(tf.float32, [None, REWARD_NUM-1], name='ph_Rew')

        A = tf.placeholder(tf.float32, ac_shape, name='action') #obs   train_model.pdtype.sample_placeholder([None])

        with tf.variable_scope(model_name, reuse=reuse):
            # x_feature = self.create_obs_feature_net(X, is_training, reuse=False)
            state = tf.concat([X, X_NEXT], axis=1)  
            state_f = self.create_obs_feature_net(state, is_training, units=[256,256,128], reuse=False)
            x_next_feature = self.dense_layer(state_f, 128, None)

            with tf.variable_scope('actor', reuse=reuse):
                h2_act = self.create_obs_feature_net(X, is_training, reuse=False)
                pi = self.dense_layer(h2_act, actdim, None) #fc(h2_act, 'pi', actdim, init_scale=0.01)
                logstd = tf.get_variable(name="logstd", shape=[1, actdim],
                    initializer=tf.zeros_initializer())    

            with tf.variable_scope('value', reuse=reuse): 
                h2_val = self.create_obs_feature_net(X, is_training, reuse=False)
                vf_1 = self.dense_layer(h2_val, 1, None) #fc(h2_val, 'vf', REWARD_NUM-1)
                vf_c = self.dense_layer(h2_val, 1, None) #fc(h2_val, 'vf', REWARD_NUM-1)

                vf = tf.concat([vf_1, vf_c], axis=1)      

            with tf.variable_scope('forward_dynamic', reuse=reuse):   
                # x_next_feature = X_NEXT 

                state_action_state = tf.concat([X, X_NEXT], axis=1)     
                h2_fd = self.create_obs_feature_net(state_action_state, is_training, units=[256, 256, 128], reuse=False, use_bn = False)
                # x_next_feature_pred = self.dense_layer(h2_fd, 32, 'next_state_f', None) #fc(h2_fd, 'next_state_f', 32)
                x_next_feature_pred = self.dense_layer(h2_fd, 128, None)

        pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pdparam)

        a0 = self.pd.sample()
        a1 = self.pd.mode()
        neglogp0 = self.pd.neglogp(a0)

        pred_error = tf.sqrt(tf.reduce_sum(tf.square(x_next_feature_pred - x_next_feature), 1))

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            a = np.clip(a, -2, 2)
            return a, v, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        def step_max(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a1, vf, neglogp0], {X:ob})
            # a = np.clip(a, -1, 1)
            return a, v, neglogp

        def get_state_action_prediction(ob, a, ob_, *_args, **_kwargs):
            return sess.run(x_next_feature_pred, {X:ob, A:a, X_NEXT:ob_})

        def get_state_feature(ob, a, ob_, *_args, **_kwargs):
            return sess.run(x_next_feature, {X:ob, A:a, X_NEXT:ob_})

        def get_pred_error(ob, a, ob_, *_args, **_kwargs):
            return sess.run(pred_error, {X:ob, A:a, X_NEXT:ob_})

        self.X = X
        self.X_NEXT = X_NEXT
        self.is_training = is_training
        self.A = A
        self.pi = pi
        self.std = self.pd.std
        self.vf = vf
        self.step = step
        self.step_max = step_max
        self.value = value
        self.mean = a1

        # self.x_feature = x_feature
        self.x_next_feature = x_next_feature
        self.x_next_feature_pred = x_next_feature_pred   
        self.state_action_pred = get_state_action_prediction
        self.state_feature = get_state_feature
        self.pred_error = get_pred_error
