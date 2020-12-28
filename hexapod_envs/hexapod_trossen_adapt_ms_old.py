import numpy as np
import mujoco_py
import hexapod_envs.my_utils as my_utils
import time
import os
from math import sqrt, acos, fabs
#from src.envs.hexapod_terrain_env.hf_gen import ManualGen, EvoGen, HMGen
import random
import string

import gym
from gym import spaces
from gym.utils import seeding

from gym.envs.mujoco.ant_v3 import AntEnv
from gym.envs.mujoco import mujoco_env
from gym import utils

class PhantomX(mujoco_env.MujocoEnv, utils.EzPickle):
    MODELPATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets/hexapod_trossen_flat_ms.xml")
    
    def __init__(self,
                 xml_file='hexapod_trossen_flat_ms.xml',
                 ctrl_cost_weight=0.5,
                 contact_cost_weight=25e-5,
                 healthy_reward=0.,
                 terminate_when_unhealthy=True,
                 healthy_z_range=(0.2, 1.0),
                 contact_force_range=(-1.0, 1.0),
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True):
        utils.EzPickle.__init__(**locals())
        
        self.leg_list = ["coxa_fl_geom","coxa_fr_geom","coxa_rr_geom","coxa_rl_geom","coxa_mr_geom","coxa_ml_geom"]

        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        
        self.ctrl_cost_weight = self._ctrl_cost_weight
        self.contact_cost_weight = self._contact_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._contact_force_range = contact_force_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        self.modelpath = os.path.join(os.path.dirname(__file__), 'assets', xml_file)
        
        #self.max_steps = 1000
        self.mem_dim = 0
        self.cumulative_environment_reward = None

        self.joints_rads_low = np.array([-0.6, -1., -1.] * 6)
        self.joints_rads_high = np.array([0.6, 0.3, 1.] * 6)
        self.joints_rads_diff = self.joints_rads_high - self.joints_rads_low

        print("Trossen hexapod")

        # Reset env variables
        self.step_ctr = 0
        self.dead_leg_sums = [0,0,0,0,0,0]

        self.modelpath = PhantomX.MODELPATH
        print(self.modelpath)
        self.max_steps = 1000
        self.mem_dim = 0
        self.cumulative_environment_reward = 0 #None

        mujoco_env.MujocoEnv.__init__(self, self.modelpath, 1)

        #self.model = mujoco_py.load_model_from_path(self.modelpath)
        #print(self.modelpath)
        #self.sim = mujoco_py.MjSim(self.model)

#        self.model.opt.timestep = 0.02
        # Environent inner parameters
 #       self.viewer = None

        # Environment dimensions
        self.q_dim = self.sim.get_state().qpos.shape[0]
        self.qvel_dim = self.sim.get_state().qvel.shape[0]

        self.obs_dim = 30 + self.mem_dim
        self.act_dim = self.sim.data.actuator_length.shape[0] + self.mem_dim

        #self.observation_space = spaces.Box(low=-1, high=1, shape=(self.obs_dim,))
        #self.action_space = spaces.Box(low=-1, high=1, shape=(self.act_dim,))

        # Reset env variables
        self.step_ctr = 0
      #  self.dead_leg_sums = [0,0,0,0,0,0]

        #self.envgen = ManualGen(12)
        #self.envgen = HMGen()
        #self.envgen = EvoGen(12)
        self.episodes = 0

        self.reset()

    def setupcam(self):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = self.model.stat.extent * 1.3
        self.viewer.cam.lookat[0] = -0.1
        self.viewer.cam.lookat[1] = 0
        self.viewer.cam.lookat[2] = 0.5
        self.viewer.cam.elevation = -20


    def scale_action(self, action):
        return (np.array(action) * 0.5 + 0.5) * self.joints_rads_diff + self.joints_rads_low


    def get_obs(self):
        qpos = self.sim.get_state().qpos.tolist()
        qvel = self.sim.get_state().qvel.tolist()
        a = qpos + qvel
        return np.asarray(a, dtype=np.float32)


    def get_obs_dict(self):
        od = {}
        # Intrinsic parameters
        for j in self.sim.model.joint_names:
            od[j + "_pos"] = self.sim.data.get_joint_qpos(j)
            od[j + "_vel"] = self.sim.data.get_joint_qvel(j)

        # Contacts:
        od['contacts'] = (np.abs(np.array(self.sim.data.cfrc_ext[[4, 7, 10, 13, 16, 19]])).sum(axis=1) > 0.05).astype(np.float32)
        #print(od['contacts'])
        #od['contacts'] = np.zeros(6)
        return od


    def get_state(self):
        return self.sim.get_state()


    def set_state(self, qpos, qvel=None):
        qvel = np.zeros(self.q_dim) if qvel is None else qvel
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()


    def render(self, close=False):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)

        self.viewer.render()

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = (np.isfinite(state).all() and min_z <= state[2] <= max_z)
        return is_healthy

    @property
    def done(self):
        done = (not self.is_healthy
                if self._terminate_when_unhealthy
                else False)
        return done

    def step_new(self, action):
        # From ant
        xy_position_before = self.get_body_com("torso")[:2].copy()
        
        scaled_action = self.scale_action(action)
        # For real robot: self.scale_action(action) instead of action
        self.do_simulation(scaled_action, self.frame_skip)
        #def do_simulation(self, ctrl, n_frames):
        #self.sim.data.ctrl[:] = ctrl
        #for _ in range(n_frames):
         #   self.sim.step()
        
        # From Nexapod:
        #act = ctrl
        #ctrl = self.scale_action(act)
        #else:
         #   mem = ctrl[-self.mem_dim:]
          #  act = ctrl[:-self.mem_dim]
           # ctrl = self.scale_action(act)

        #self.prev_act = np.array(act)

        #self.sim.data.ctrl[:] = ctrl
        #self.sim.forward()
        #self.sim.step()
        
        xy_position_after = self.get_body_com("torso")[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        # Reward calculation

        # use scaled action (see above)
        ctrl_cost = control_cost = 0.5 * np.sum(np.square(action)) #self.control_cost(scaled_action)
        contact_cost = 0#self.contact_cost

        forward_reward = x_velocity
        #healthy_reward = self.healthy_reward

        rewards = forward_reward #+ healthy_reward
        costs = ctrl_cost + contact_cost

        reward = rewards - costs
        done = self.done

        # Reward calc from nexapod
#        target_vel = 0.25
 #       velocity_rew = 1. / (abs(xd - target_vel) + 1.) - 1. / (target_vel + 1.)
#        r = velocity_rew * 10 - \
 #           np.square(self.sim.data.actuator_force).mean() * 0.0001 - \
  #          np.abs(roll) * 0.1 - \
   #         np.square(pitch) * 0.1 - \
    #        np.square(yaw) * .1 - \
     #       np.square(y) * 0.1 - \
      #      np.square(zd) * 0.01
       # r = np.clip(r, -2, 2)
        
        #ctrl_cost = self.control_cost(action)
#        control_cost = 0.5 * np.sum(np.square(ctrl))
 #       contact_cost = 0.
        #contact_cost = self.contact_cost
        # From ant
#        forward_reward = x_velocity
        healthy_reward = 0. #self.healthy_reward
        # From ant
#        rewards = forward_reward #+ healthy_reward
 #       costs = control_cost + contact_cost
  #      reward = rewards - costs
   #     r = reward
        #self.cumulative_environment_reward += r
        # Reevaluate termination condition
        #done = self.step_ctr > self.max_steps # or abs(y) > 0.3 or x < -0.2 or abs(yaw) > 0.8
#        done = self.done
        
        observation = self._get_obs()
        
        info = {
            'reward_forward': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'reward_contact': -contact_cost,
            'reward_survive': healthy_reward,

            'x_position': xy_position_after[0],
            'y_position': xy_position_after[1],
            'distance_from_origin': np.linalg.norm(xy_position_after, ord=2),

            'x_velocity': x_velocity,
            'y_velocity': y_velocity,
            'forward_reward': forward_reward,
        }

        return np.clip(observation, -1., 1.), reward, done, info

    @property
    def contact_forces(self):
        raw_contact_forces = self.sim.data.cfrc_ext
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    def _get_obs(self):
        # From nexapod:
        # obs = self.get_obs() 
#               qpos = self.sim.get_state().qpos.tolist()
#               qvel = self.sim.get_state().qvel.tolist()
#               a = qpos + qvel
#               return np.asarray(a, dtype=np.float32)
#        obs_dict = self.get_obs_dict()
        # Angle deviation
#        x, y, z, qw, qx, qy, qz = obs[:7]
#        xd, yd, zd, _, _, _ = self.sim.get_state().qvel.tolist()[:6]
 #       angle = 2 * acos(qw)
#        roll, pitch, yaw = my_utils.quat_to_rpy((qw, qx, qy, qz))       
#        obs = np.concatenate([np.array(self.sim.get_state().qpos.tolist()[3:]),
 #                             [xd, yd],
  #                            obs_dict["contacts"]])
        # if np.random.rand() < self.dead_leg_prob:
        #     idx = np.random.randint(0,6)
        #     self.dead_leg_vector[idx] = 1
        #     self.dead_leg_sums[idx] += 1
        #     self.model.geom_rgba[self.model._geom_name2id[self.leg_list[idx]]] = [1, 0, 0, 1]
        #     self.dead_leg_prob = 0.
#        return np.clip(obs, -1., 1.), r, done, obs_dict
    
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        contact_force = self.contact_forces.flat.copy()
        #print(contact_force)

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        observations = np.concatenate((position, velocity, contact_force))

        return observations

    def step(self, ctrl):
        # From ant
        xy_position_before = self.get_body_com("torso")[:2].copy()

        act = ctrl
        ctrl = self.scale_action(act)


        self.sim.data.ctrl[:] = ctrl
        self.sim.forward()
        self.sim.step()

        obs = self.get_obs()
        obs_dict = self.get_obs_dict()

        # Angle deviation
        x, y, z, qw, qx, qy, qz = obs[:7]

        xd, yd, zd, _, _, _ = self.sim.get_state().qvel.tolist()[:6]
        angle = 2 * acos(qw)

        roll, pitch, yaw = my_utils.quat_to_rpy((qw, qx, qy, qz))

        # Reward conditions
        target_vel = 0.25
        velocity_rew = 1. / (abs(xd - target_vel) + 1.) - 1. / (target_vel + 1.)

        r = velocity_rew * 10 - \
            np.square(self.sim.data.actuator_force).mean() * 0.0001 - \
            np.abs(roll) * 0.1 - \
            np.square(pitch) * 0.1 - \
            np.square(yaw) * .1 - \
            np.square(y) * 0.1 - \
            np.square(zd) * 0.01
        r = np.clip(r, -2, 2)
        
        # From ant
        xy_position_after = self.get_body_com("torso")[:2].copy()
        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity
        #x_velocity = xd / self.model.opt.timestep
        
        # From ant
        #ctrl_cost = self.control_cost(action)
        control_cost = 0.5 * np.sum(np.square(ctrl))
        contact_cost = 0.
        #contact_cost = self.contact_cost

        # From ant
        forward_reward = x_velocity
        #healthy_reward = self.healthy_reward
        
        # From ant
        rewards = forward_reward #+ healthy_reward
        costs = control_cost + contact_cost
        reward = rewards - costs

        # Reevaluate termination condition
        done = self.done
        done = self.step_ctr > self.max_steps # or abs(y) > 0.3 or x < -0.2 or abs(yaw) > 0.8

        obs = np.concatenate([np.array(self.sim.get_state().qpos.tolist()[3:]),
                              [xd, yd],
                              obs_dict["contacts"]])

        return np.clip(obs, -1., 1.), reward, done, obs_dict

    def step_old(self, ctrl):
        # From ant
        #xy_position_before = self.get_body_com("torso")[:2].copy()
        
        # Mute appropriate leg joints
        #for i in range(6):
         #   if self.dead_leg_vector[i] == 1:
          #      ctrl[i * 3:i * 3 + 3] = np.zeros(3) #np.random.randn(3) * 0.1

        if self.mem_dim == 0:
            mem = np.zeros(0)
            act = ctrl
            ctrl = self.scale_action(act)
        else:
            mem = ctrl[-self.mem_dim:]
            act = ctrl[:-self.mem_dim]
            ctrl = self.scale_action(act)

        self.prev_act = np.array(act)

        self.sim.data.ctrl[:] = ctrl
        self.sim.forward()
        self.sim.step()
        self.step_ctr += 1

        obs = self.get_obs()
        obs_dict = self.get_obs_dict()

        # Angle deviation
        x, y, z, qw, qx, qy, qz = obs[:7]

        xd, yd, zd, _, _, _ = self.sim.get_state().qvel.tolist()[:6]
        angle = 2 * acos(qw)

        roll, pitch, yaw = my_utils.quat_to_rpy((qw, qx, qy, qz))

        # Reward conditions
        target_vel = 0.25
        velocity_rew = 1. / (abs(xd - target_vel) + 1.) - 1. / (target_vel + 1.)

        r = velocity_rew * 10 - \
            np.square(self.sim.data.actuator_force).mean() * 0.0001 - \
            np.abs(roll) * 0.1 - \
            np.square(pitch) * 0.1 - \
            np.square(yaw) * .1 - \
            np.square(y) * 0.1 - \
            np.square(zd) * 0.01
        r = np.clip(r, -2, 2)
        
        # From ant
        #xy_position_after = self.get_body_com("torso")[:2].copy()
        #xy_velocity = (xy_position_after - xy_position_before) / self.dt
        #x_velocity, y_velocity = xy_velocity
        x_velocity = xd / self.model.opt.timestep
        
        # From ant
        #ctrl_cost = self.control_cost(action)
        control_cost = 0.5 * np.sum(np.square(ctrl))
        contact_cost = 0.
        #contact_cost = self.contact_cost

        # From ant
        forward_reward = x_velocity
        #healthy_reward = self.healthy_reward
        
        # From ant
        rewards = forward_reward #+ healthy_reward
        costs = control_cost + contact_cost
        reward = rewards - costs
        r = reward

        self.cumulative_environment_reward += r

        # Reevaluate termination condition
        done = self.step_ctr > self.max_steps # or abs(y) > 0.3 or x < -0.2 or abs(yaw) > 0.8

        obs = np.concatenate([np.array(self.sim.get_state().qpos.tolist()[3:]),
                              [xd, yd],
                              obs_dict["contacts"],
                              mem])

        # if np.random.rand() < self.dead_leg_prob:
        #     idx = np.random.randint(0,6)
        #     self.dead_leg_vector[idx] = 1
        #     self.dead_leg_sums[idx] += 1
        #     self.model.geom_rgba[self.model._geom_name2id[self.leg_list[idx]]] = [1, 0, 0, 1]
        #     self.dead_leg_prob = 0.

        return np.clip(obs, -1., 1.), r, done, obs_dict


    def reset(self):

        self.cumulative_environment_reward = 0
        self.dead_leg_prob = 0.004
        self.dead_leg_vector = [0, 0, 0, 0, 0, 0]
        self.step_ctr = 0

        for i in range(6):
#            if self.dead_leg_vector[i] ==0:
            self.model.geom_rgba[self.model._geom_name2id[self.leg_list[i]]] = [0.0, 0.6, 0.4, 1]
            #else:
             #   self.model.geom_rgba[self.model._geom_name2id[self.leg_list[i]]] = [1, 0, 0, 1]

        # Sample initial configuration
        init_q = np.zeros(self.q_dim, dtype=np.float32)
        init_q[0] = 0.05
        init_q[1] = 0
        init_q[2] = 0.15
        init_qvel = np.random.randn(self.qvel_dim).astype(np.float32) * 0.1

        # Set environment state
        self.set_state(init_q, init_qvel)

        #self.prev_act = np.zeros((self.act_dim - self.mem_dim))

        obs, _, _, _ = self.step(np.zeros(self.act_dim))

        return obs


    def demo(self):
        self.reset()
        for i in range(1000):
            #self.step(np.random.randn(self.act_dim))
            for i in range(100):
                self.step(np.zeros((self.act_dim)))
                self.render()
            for i in range(100):
                self.step(np.ones((self.act_dim)) * 1)
                self.render()
            for i in range(100):
                self.step(np.ones((self.act_dim)) * -1)
                self.render()


    def info(self):
        self.reset()
        for i in range(100):
            a = np.ones((self.act_dim)) * 0
            obs, _, _, _ = self.step(a)
            print(obs[[3, 4, 5]])
            self.render()
            time.sleep(0.01)

        print("-------------------------------------------")
        print("-------------------------------------------")


    def test(self, policy):
        #self.envgen.load()
        for i in range(100):
            obs = self.reset()
            cr = 0
            for j in range(self.max_steps):
                action = policy(my_utils.to_tensor(obs, True)).detach()
                #print(action[0, :-self.mem_dim])
                obs, r, done, od, = self.step(action[0].numpy())
                cr += r
                time.sleep(0.001)
                self.render()
            print("Total episode reward: {}".format(cr))


    def test_recurrent(self, policy):
        self.reset()
        for i in range(100):
            obs = self.reset()
            h = None
            cr = 0
            for j in range(self.max_steps):
                action, h = policy((my_utils.to_tensor(obs, True).unsqueeze(0), h))
                obs, r, done, od, = self.step(action[0,0].detach().numpy())
                cr += r
                time.sleep(0.001)
                self.render()
            print("Total episode reward: {}".format(cr))


if __name__ == "__main__":
    ant = PhantomX(animate=True)
    print(ant.obs_dim)
    print(ant.act_dim)
    ant.demo()