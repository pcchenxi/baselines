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
    def create_obs_feature_net(self, obs, reuse=False):
        with tf.variable_scope('obs_feature', reuse=reuse):        
            h1_obsf = activ(fc(obs, 'obs_f_h1', nh=64, init_scale=np.sqrt(2)))
            h2_obsf = activ(fc(h1_obsf, 'obs_f_h2', nh=32, init_scale=np.sqrt(2)))
            obs_feature = fc(h2_obsf, 'obs_feature', 32)
        return obs_feature

    def __init__(self, sess, ob_space, ac_space, REWARD_NUM, reuse=False, model_name="model"): #pylint: disable=W0613
        # with tf.device('/device:GPU:0'):
        ob_shape = (None,) + ob_space.shape
        ac_shape = (None,) + ac_space.shape
        print('ob_space.shape', ob_space.shape)

        actdim = ac_space.shape[0]
        X = tf.placeholder(tf.float32, ob_shape, name='Ob') #obs
        X_NEXT = tf.placeholder(tf.float32, ob_shape, name='Ob_next') #obs

        A = tf.placeholder(tf.float32, ac_shape, name='action') #obs   train_model.pdtype.sample_placeholder([None])
        
        with tf.variable_scope(model_name, reuse=reuse):
            with tf.variable_scope('actor', reuse=reuse):
                h1_act = activ(fc(X, 'pi_fc1', nh=256, init_scale=np.sqrt(2)))
                h2_act = activ(fc(h1_act, 'pi_fc2', nh=128, init_scale=np.sqrt(2)))
                pi = fc(h2_act, 'pi', actdim, init_scale=0.01)
                logstd = tf.get_variable(name="logstd", shape=[1, actdim],
                    initializer=tf.zeros_initializer())    

            with tf.variable_scope('value', reuse=reuse):        
                h1_val = activ(fc(X, 'vf_fc1', nh=128, init_scale=np.sqrt(2)))
                h2_val = activ(fc(h1_val, 'vf_fc2', nh=64, init_scale=np.sqrt(2)))
                vf_1 = fc(h2_val, 'vf', REWARD_NUM-1)
                vf_ri = fc(h2_val, 'vf_ri', 1)

                vf = tf.concat([vf_1, vf_ri], axis=1)      

            with tf.variable_scope('inverse_dynamic', reuse=reuse): 
                x_feature = self.create_obs_feature_net(X, reuse=False)
                x_next_feature = self.create_obs_feature_net(X_NEXT, reuse=True)
                combined_state_f = tf.concat([x_feature, x_next_feature], axis=1)

                rp_h1 = activ(fc(combined_state_f, 'rp_h1', nh=64, init_scale=np.sqrt(2)))
                rp_h2 = activ(fc(rp_h1, 'rp_h2', nh=64, init_scale=np.sqrt(2)))
                inverse_dynamic_pred = fc(rp_h2, 'inverse_pred', REWARD_NUM-1)

            with tf.variable_scope('forward_dynamic', reuse=reuse):   
                state_action = tf.concat([x_feature, A], axis=1)     
                h1_fd = activ(fc(state_action, 'fd_h1', nh=64, init_scale=np.sqrt(2)))
                h2_fd = activ(fc(h1_fd, 'fd_h2', nh=32, init_scale=np.sqrt(2)))
                x_next_feature_pred = fc(h2_fd, 'next_state_f', 32)


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

        def get_state_action_prediction(ob, a, *_args, **_kwargs):
            return sess.run(x_next_feature_pred, {X:ob, A:a})

        def get_state_feature(ob, *_args, **_kwargs):
            return sess.run(x_feature, {X:ob})

        def step_max(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a1, vf, neglogp0], {X:ob})
            # a = np.clip(a, -1, 1)
            return a, v, self.initial_state, neglogp

        self.X = X
        self.X_NEXT = X_NEXT
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

