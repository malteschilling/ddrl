from gym.envs.mujoco.ant_v3 import AntEnv
import numpy as np
import os
from scipy import ndimage
from scipy.signal import convolve2d

DEFAULT_CAMERA_CONFIG = {
    'distance': 10.0,
    'type': 1,
    'trackbodyid': 1,
    'elevation': -5.0,
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
    #print("Smoothness set to: ", smoothness)

class QuAntruped_TVel_Env(AntEnv):
    """ Environment with a quadruped walker - derived from the ant_v3 environment
        
        Uses a different observation space compared to the ant environment (less inputs).
        Per default, healthy reward is turned of (unnecessary).
        
        The environment introduces a heightfield which allows to test or train
        the system in uneven terrain (generating new heightfields has to be explicitly
        called, ideally before a reset of the system).
    """ 
    def __init__(self, ctrl_cost_weight=0.5, contact_cost_weight=5e-4, healthy_reward=0., hf_smoothness=1.):
        # Agent is rewarded for reaching a given target velocity.
        self.target_vel = np.array([1.]) 
        
        # Some statistics collected during running, for debugging.
        self.start_pos = None
        self.step_counter = 0
        self.vel_rewards = 0.
        self.sum_rewards = 0.
        self.ctrl_costs = 0.
        
        self.max_steps = 1000
        
        super().__init__(xml_file=os.path.join(os.path.dirname(__file__), 'assets','ant_hfield.xml'), ctrl_cost_weight=ctrl_cost_weight, contact_cost_weight=contact_cost_weight)
        self.ctrl_cost_weight = self._ctrl_cost_weight
        self.contact_cost_weight = self._contact_cost_weight
        
        # Heightfield
        self.hf_smoothness = hf_smoothness
        self.hf_bump_scale = 2.
        create_new_hfield(self.model, self.hf_smoothness, self.hf_bump_scale)
        
        # Otherwise when learning from scratch might abort
        # This allows for more collisions.
        # Should be moved to xml file.
        self.model.nconmax = 500 
        self.model.njmax = 2000
        
        self.start_pos = self.sim.data.qpos[0].copy()


    def reset(self):
        obs = super().reset()
        self.start_pos = self.sim.data.qpos[0] #self.get_body_com("torso")[:2].copy()
        self.step_counter = 0
        self.vel_rewards = 0.
        self.sum_rewards = 0.
        self.ctrl_costs = 0.
        return obs
        
#    @property
 #   def healthy_reward(self):
  #      return 0.
  
    def set_target_velocity(self, t_vel):
        self.target_vel = np.array([t_vel])
  
    def create_new_random_hfield(self):
        create_new_hfield(self.model, self.hf_smoothness, self.hf_bump_scale)

    def step(self, action):
        xy_position_before = self.get_body_com("torso")[:2].copy()
        # Call simulation to make a step (frame_skip steps)
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()

        # Calculate velocity for reward.
        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity
        
        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost

        forward_reward = (1. + 1./self.target_vel[0]) * (1. / (np.abs(x_velocity - self.target_vel[0]) + 1.) - 1. / (self.target_vel[0] + 1.))
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward
        costs = ctrl_cost + contact_cost

        reward = rewards - costs
        done = self.done
        
        self.ctrl_costs += ctrl_cost
        self.vel_rewards += forward_reward
        self.sum_rewards += rewards
        self.step_counter += 1
        
        # Print results from an episode.
        if done or self.step_counter == self.max_steps:
            distance = (self.sim.data.qpos[0] - self.start_pos)# / (self.step_counter * self.dt)
            print("Quantruped episode: ", distance, " / vel: : ",\
                (distance/ (self.step_counter * self.dt)), \
                x_velocity, self.target_vel[0], self.vel_rewards, \
                " / ctrl: ", self.ctrl_costs, " / sum rew: ", self.sum_rewards, self.step_counter)
        
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

    def _get_obs(self):
        """ 
        Observation space for the QuAntruped model.
        
        Following observation spaces are used: 
        * position information
        * velocity information
        * passive forces acting on the joints
        * last control signal
        
        Unfortunately, the numbering schemes are different for the legs depending on the
        specific case: actions and measurements use each their own scheme.
        
        For actions (action_space and .sim.data.ctrl) ordering is 
        (front means x direction, in rendering moving to the right; rewarded direction)
            Front right: 0 = hip joint - positive counterclockwise (from top view), 
                         1 = knee joint - negative is up
            Front left: 2 - pos. ccw., 3 - neg. is up
            Hind left: 4 - pos. ccw., 5 - pos. is up
            Hind right: 6 - pos. ccw., 7 - pos. is up
        
        For measured observations (basically everything starting with a q) ordering is:
            FL: 0, 1
            HL: 2, 3
            HR: 4, 5
            FR: 6, 7
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

        observations = np.concatenate((position, velocity, joint_sensor_forces, last_control, self.target_vel))#, last_control)) #, contact_force))

        return observations
    
    def set_hf_parameter(self, smoothness, bump_scale=None):
        self.hf_smoothness = smoothness
        if bump_scale:
            self.hf_bump_scale = bump_scale
           
    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                 getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)