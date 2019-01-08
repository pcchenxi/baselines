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
from baselines.ppo_goal.common_functions import *


import gym, roboschool, sys, os, gym_hockeypuck
import pyglet, pyglet.window as pw, pyglet.window.key as pwk
from pyglet import gl
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from baselines.ppo_goal.policies import MlpPolicy_Goal as MlpPolicy

import gym.spaces, gym.utils, gym.utils.seeding
from roboschool.scene_abstract import cpp_household

#
# This opens a third-party window (not test window), shows rendered chase camera, allows to control humanoid
# using keyboard (in a different way)
#

class PygletInteractiveWindow(pw.Window):
    def __init__(self, env):
        pw.Window.__init__(self, width=600, height=400, vsync=False, resizable=True)
        self.theta = 0
        self.still_open = True

        @self.event
        def on_close():
            self.still_open = False

        @self.event
        def on_resize(width, height):
            self.win_w = width
            self.win_h = height

        self.keys = {}
        self.human_pause = False
        self.human_done = False

    def imshow(self, arr):
        H, W, C = arr.shape
        assert C==3
        image = pyglet.image.ImageData(W, H, 'RGB', arr.tobytes(), pitch=W*-3)
        self.clear()
        self.switch_to()
        self.dispatch_events()
        texture = image.get_texture()
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        texture.width  = W
        texture.height = H
        texture.blit(0, 0, width=self.win_w, height=self.win_h)
        time.sleep(0.005)
        self.flip()

    def on_key_press(self, key, modifiers):
        self.keys[key] = +1
        if key==pwk.ESCAPE: self.still_open = False

    def on_key_release(self, key, modifiers):
        self.keys[key] = 0

    def each_frame(self):
        self.theta += 0.05 * (self.keys.get(pwk.LEFT, 0) - self.keys.get(pwk.RIGHT, 0))

def HUD(scene, r):
    scene.cpp_world.test_window_history_advance()
    # scene.cpp_world.test_window_observations(s.tolist())
    # scene.cpp_world.test_window_actions(a.tolist())
    scene.cpp_world.test_window_rewards(r)

def demo_run(model_index):
    print('run demo')
    # env = gym.make("RoboschoolHumanoid-v1")
    # env = gym.make("RoboschoolAnt-v1")
    # env = gym.make("RoboschoolHalfCheetah-v1")
    # env = gym.make("LunarLanderContinuous-v2")
    env = gym.make("RoboschoolReacher-v1")

    policy = MlpPolicy
    ob_space = env.observation_space
    ac_space = env.action_space

    make_model = lambda model_name, need_summary : Model(model_name=model_name, policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=1, nbatch_train=1000,
                    nsteps=1000, ent_coef=0, vf_coef=0, lr = 0,
                    max_grad_norm=0, need_summary = False)

    model = make_model('model'+str(model_index), need_summary = False)
    # model_index = 2
    name = 'reacher'
    # model_path = '/home/xi/workspace/model/exp_meta_morl/' + str(model_index)
    # model_path = '/home/xi/workspace/model/exp_mp_morl/' + str(model_index)

    # model_path = '/home/xi/workspace/exp_models/exp_' + name + '/meta_finetune/' + str(model_index-1)
    model_path = '/home/xi/workspace/exp_models/exp_' + name + '/mp_train/' + str(model_index-1)
    model.load(model_path)

    random_ws = np.load('/home/xi/workspace/weights_half.npy')

    control_me = PygletInteractiveWindow(env.unwrapped)
    env.reset()
    eu = env.ustates
    obs = env.reset()

    states = model.initial_state

    print(random_ws[model_index])
    for _ in range(500):
        # actions, values, states, neglogpacs = model.step_max([obs], states, False)
        actions, values, states, neglogpacs = model.step([obs], states, False)
        real_value_old = model_old.value([obs])[0,0]
        # x, y, z = eu.body_xyz
        # eu.walk_target_x = x + 100.1*np.cos(control_me.theta)   # 1.0 or less will trigger flag reposition by env itself
        # eu.walk_target_y = y + 100.1*np.sin(control_me.theta)
        # eu.flag = eu.scene.cpp_world.debug_sphere(eu.walk_target_x, eu.walk_target_y, 0.2, 0.2, 0xFF8080)
        # eu.flag_timeout = 100500

        obs, r, done, _ = env.step(actions[0])
        img = env.render("rgb_array")
        control_me.imshow(img)
        control_me.each_frame()
        if done:
            break       
    env.close() 
        # if control_me.still_open==False: break



def run_test():
    # env = gym.make("hockeypuck-v0")
    # env.seed()
    # RoboschoolHumanoidFlagrunHarder  FetchSlide FetchPush FetchPickAndPlace
    env = gym.make("FetchPush-v1")

    policy = MlpPolicy
    ob_spaces = env.observation_space.spaces.items()
    ac_space = env.action_space

    index = 0
    for a in ob_spaces:
        if index == 2:
            ob_space = a[1]
        else:
            goal_space = a[1]
        index += 1

    # print(ob_space.shape, ac_space)

    make_model = lambda model_name, need_summary : Model(model_name=model_name, policy=policy, ob_space=ob_space, ac_space=ac_space, goal_space=goal_space, nbatch_act=1, nbatch_train=1000,
                    nsteps=1000, ent_coef=0, vf_coef=0, lr = 0,
                    max_grad_norm=0, need_summary = False)

    model = make_model('model', need_summary = False)
    model_old = make_model('model_old', need_summary = False)

    model_path = '/home/xi/workspace/model/checkpoints/0'
    # model_path = '/home/xi/workspace/model/log/exp/puck/normal/c_driven/checkpoint/1000'
    
    model.load(model_path)

    model_path = '/home/xi/workspace/model/checkpoints/1'    
    model_old.load(model_path)

    full_obs = env.reset()

    sim_data = env.get_sim_data()
    sim_goal = full_obs['achieved_goal']
    # env.set_sim_data(sim_data, sim_goal)
    # print('get data', sim_data)
    print('goal', sim_data[1], full_obs['desired_goal'])

    # print(env.get_sim_data())

    runner = Runner(env=env, model=model, nsteps=4096, gamma=0.99, lam=0.99)
    runner.init_task_pool(15, model_old, render=True)
    # obs, returns, dones, actions, values, advs_ori, rewards, neglogpacs, epinfos, ret = runner.run(int(300), is_test=False, render=True) #pylint: disable=E0632
    
    for j in range(100):    
        print('reset')
        full_obs = env.reset()   
        task_index = np.random.choice(len(runner.tasks), 1, p=runner.tasks_vlaue)[0]
        print(task_index, runner.tasks_vlaue[task_index], len(runner.tasks))
        task = runner.tasks[task_index]    
        full_obs = env.set_sim_data(task[0], task[1])
        obs = np.concatenate((full_obs['observation'], full_obs['desired_goal']), axis=0) 

        for i in range(50):
            actions, values, neglogpacs = model.step([obs])
            full_obs, r, done, info = env.step(actions[0])

            obs = np.concatenate((full_obs['observation'], full_obs['desired_goal']), axis=0) 
            real_value_old = model_old.value([obs])[0,0]

            env.render('human')
            print(r, values, real_value_old)
            if done:
                # full_obs = env.reset()  
                # full_obs = env.set_sim_data(sim_data, sim_goal)     
                break       
    env.close() 
        
if __name__=="__main__":
    # demo_list = np.array([7, 8, 10, 18, 14, 24, 27])-1
    # index = 5
    # random_ws = np.load('/home/xi/workspace/weights_half.npy')

    # random_ws = np.around(random_ws, decimals=2 )
    # print(random_ws[demo_list])

    # demo_run(demo_list[index-1])


    run_test()