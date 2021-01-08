from gym.envs.mujoco.ant_v3 import AntEnv
import numpy as np
import os
from scipy import ndimage
from scipy.signal import convolve2d

DEFAULT_CAMERA_CONFIG = {
    'distance': 2.,
    'type': 1,
    'trackbodyid': 1,
    'elevation': -12.0,
}

def create_new_hfield(mj_model, smoothness = 0.15, bump_scale=2.):
    # Generation of the shape of the height field is taken from the dm_control suite,
    # see dm_control/suite/quadruped.py in the escape task (but we don't use the bowl shape).
    # Their parameters are TERRAIN_SMOOTHNESS = 0.15  # 0.0: maximally bumpy; 1.0: completely smooth.
    # and TERRAIN_BUMP_SCALE = 2  # Spatial scale of terrain bumps (in meters). 
    res = mj_model.hfield_ncol[0]
    row_grid, col_grid = np.ogrid[-1:1:res*1j, -1:1:res*1j]
    # Random smooth bumps.
    terrain_size = 2 * mj_model.hfield_size[0, 0]
    bump_res = int(terrain_size / bump_scale)
    bumps = np.random.uniform(smoothness, 1, (bump_res, bump_res))
    smooth_bumps = ndimage.zoom(bumps, res / float(bump_res))
    # Terrain is elementwise product.
    hfield = (smooth_bumps - np.min(smooth_bumps))[0:mj_model.hfield_nrow[0],0:mj_model.hfield_ncol[0]]
    # Clears a patch shaped like box, assuming robot is placed in center of hfield.
    # Function was implemented in an old rllab version.
    h_center = int(0.5 * hfield.shape[0])
    w_center = int(0.5 * hfield.shape[1])
    patch_size = 8
    fromrow, torow = h_center - int(0.5*patch_size), h_center + int(0.5*patch_size)
    fromcol, tocol = w_center - int(0.5*patch_size), w_center + int(0.5*patch_size)
    # convolve to smoothen edges somewhat, in case hills were cut off
    K = np.ones((patch_size,patch_size)) / patch_size**2
    s = convolve2d(hfield[fromrow-(patch_size-1):torow+(patch_size-1), fromcol-(patch_size-1):tocol+(patch_size-1)], K, mode='same', boundary='symm')
    hfield[fromrow-(patch_size-1):torow+(patch_size-1), fromcol-(patch_size-1):tocol+(patch_size-1)] = s
    # Last, we lower the hfield so that the centre aligns at zero height
    # (importantly, we use a constant offset of -0.5 for rendering purposes)
    #print(np.min(hfield), np.max(hfield))
    hfield = hfield - np.max(hfield[fromrow:torow, fromcol:tocol])
    mj_model.hfield_data[:] = hfield.ravel()
    print("Smoothness set to: ", smoothness)

class SixLeggedEnv(AntEnv):
    """ Environment with a quadruped walker - derived from the ant_v3 environment
        
        Uses a different observation space compared to the ant environment (less inputs).
        Per default, healthy reward is turned of (unnecessary).
        
        The environment introduces a heightfield which allows to test or train
        the system in uneven terrain (generating new heightfields has to be explicitly
        called, ideally before a reset of the system).
    """ 
    def __init__(self, ctrl_cost_weight=0.5, contact_cost_weight=5e-4, healthy_reward=0., hf_smoothness=1.):
        self.step_counter = 0
        self.ctrl_costs = 0.
        self.sum_rewards = 0.
        self.vel_rewards = 0.
        self.target_vel = np.array([0.16])
        super().__init__(xml_file=os.path.join(os.path.dirname(__file__), 'assets','Hexapod_PhantomX_smallJointRanges.xml'), 
            ctrl_cost_weight=ctrl_cost_weight, 
            contact_cost_weight=contact_cost_weight,
            healthy_reward=0.,
            healthy_z_range=(0.025, 1.5),
            reset_noise_scale=0.05)
        #super().__init__(ctrl_cost_weight=ctrl_cost_weight, contact_cost_weight=contact_cost_weight)
        self.ctrl_cost_weight = self._ctrl_cost_weight
        self.contact_cost_weight = self._contact_cost_weight
        self.hf_smoothness = hf_smoothness
        self.hf_bump_scale = 2.
        
        self.init_qpos[9] = 0.55
        self.init_qpos[12] = 0.55
        self.init_qpos[15] = 0.55
        self.init_qpos[18] = 0.55
        self.init_qpos[21] = 0.55
        self.init_qpos[24] = 0.55
        create_new_hfield(self.model, self.hf_smoothness, self.hf_bump_scale)
        
        # Otherwise when learning from scratch might abort
        # This allows for more collisions.
        #self.model.nconmax = 500 
        #self.model.njmax = 2000
  
#    def create_new_random_hfield(self):
#        create_new_hfield(self.model, self.hf_smoothness, self.hf_bump_scale)

#     def control_cost(self, action):
#         control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
#         return control_cost
#         
#     @property
#     def contact_cost(self):
#         contact_cost = self._contact_cost_weight * np.sum(
#             np.square(self.contact_forces))
#         return contact_cost

    def set_target_velocity(self, t_vel):
        self.target_vel = np.array([t_vel])

    def step(self, action): #setpoints):
        # From ant
        xy_position_before = self.get_body_com("torso")[:2].copy()
        
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        # Reward calculation
        # use scaled action (see above)
        ctrl_cost = self.control_cost(action) #torques
        contact_cost = self.contact_cost

        forward_reward = x_velocity #* 10 # Scaled as ant-sim env is much bigger
        #forward_reward = (1. + 1./self.target_vel[0]) * (1. / (np.abs(x_velocity - self.target_vel[0]) + 1.) - 1. / (self.target_vel[0] + 1.))
        
        healthy_reward = 0. #self.healthy_reward
        
        done = self.done
        
        if (self.calculate_torso_z_orientation() < -0.7):
            done = True
            forward_reward += (self.step_counter - 1000)
        
        rewards = forward_reward #+ healthy_reward
        costs = ctrl_cost + contact_cost

        self.ctrl_costs += ctrl_cost
        #self.contact_costs += contact_cost
        self.vel_rewards += forward_reward
        #self.healthy_rewards += healthy_reward

        reward = rewards - costs
        self.sum_rewards += reward
        #if (self.step_counter % 50 == 0):
         #   print("REW: ", reward, forward_reward, healthy_reward)
        
        self.step_counter += 1
        
        if done or self.step_counter == 1000:
            distance = self.sim.data.qpos[0]# / (self.step_counter * self.dt)
            print("Hexapod target vel episode: ", distance, \
                (distance/ (self.step_counter * self.dt)), \
                x_velocity, self.vel_rewards, \
                " / ctrl: ", self.ctrl_costs, self.ctrl_cost_weight, \
                " overall: ", self.sum_rewards, self.step_counter)
        
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

        return observation, reward, done, info

#     def step(self, action):
#         obs, reward, done, info = super().step(action)
#         
#         self.ctrl_costs += info['reward_ctrl']
#         self.sum_rewards += reward
#         self.step_counter += 1
#         
#         if done or self.step_counter == 1000:
#             distance = self.sim.data.qpos[0]
#             print("Hexapod episode: ", distance, (distance/ (self.step_counter * self.dt)), \
#                 " / ctrl: ", self.ctrl_costs, self.ctrl_cost_weight, \
#                 " overall: ", self.sum_rewards, self.step_counter)
#         return obs, reward, done, info

    def _get_obs(self):
        """ 
        Observation space for the Sixlegged model.
        
        Following observation spaces are used: 
        * position information
        * velocity information
        * passive forces acting on the joints
        * last control signal
        
        Ordering is FL, FR, ML, MR, HL, HR
        """
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        #contact_force = self.contact_forces.flat.copy()
        # Provide passive force instead -- in joint reference frame = eight dimensions
        # joint_passive_forces = self.sim.data.qfrc_passive.flat.copy()[6:]
        # Sensor measurements in the joint:
        # qfrc_unc is the sum of all forces outside constraints (passive, actuation, gravity, applied etc)
        # qfrc_constraint is the sum of all constraint forces. 
        # If you add up these two quantities you get the total force acting on each joint
        # which is what a torque sensor should measure.
        # See note in http://www.mujoco.org/forum/index.php?threads/best-way-to-represent-robots-torque-sensors.4181/
        joint_sensor_forces = self.sim.data.qfrc_unc[6:] + self.sim.data.qfrc_constraint[6:]

        # Provide actions from last time step (as used in the simulator = clipped)
        last_control = self.sim.data.ctrl.flat.copy()
        
        if self._exclude_current_positions_from_observation:
            position = position[2:]

        observations = np.concatenate((position, velocity, joint_sensor_forces, last_control))#, last_control)) #, contact_force))

        return observations
    
    def set_hf_parameter(self, smoothness, bump_scale=None):
        self.hf_smoothness = smoothness
        if bump_scale:
            self.hf_bump_scale = bump_scale
    
    def calculate_torso_z_orientation(self):
        # Calculate if model keeps upright
        # Current orientation as a matrix
        torso_orient_mat = self.sim.data.body_xmat[1].reshape(3,3)
        # Reward is projection of z axis of body onto world z-axis
        z_direction = np.matmul(torso_orient_mat, np.array([0.,0.,1.]))[2]#0. #self.healthy_reward
        return z_direction
        
    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = (np.isfinite(state).all() and min_z <= state[2] <= max_z)
        if (self.calculate_torso_z_orientation() < -0.7):
            is_healthy = False
        return is_healthy
           
    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                 getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
                
    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)
        self.set_state(qpos, qvel)

        observation = self._get_obs()

        self.step_counter = 0
        self.ctrl_costs = 0.
        self.sum_rewards = 0
        self.vel_rewards = 0.

        return observation