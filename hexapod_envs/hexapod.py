import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import os
from gym.envs.mujoco.ant_v3 import AntEnv

# DEFAULT_CAMERA_CONFIG = {
#     'distance': 4.0,
# }

DEFAULT_CAMERA_CONFIG = {
    'distance': 15.0,
    'type': 1,
    'trackbodyid': 1,
    'elevation': -20.0,
}

class HexapodEnv(AntEnv):

    MODELPATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             "assets/")

    def __init__(self,
                 xml_file='hexapod_trossen_ms.xml',
                 ctrl_cost_weight=0.5,
                 contact_cost_weight=5e-4,
                 healthy_reward=1.0,
                 terminate_when_unhealthy=True,
                 healthy_z_range=(0.2, 1.0),
                 contact_force_range=(-1.0, 1.0),
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True,
                 hf_smoothness=1.):
        #utils.EzPickle.__init__(**locals())

        super().__init__(xml_file=os.path.join(HexapodEnv.MODELPATH, xml_file), ctrl_cost_weight=ctrl_cost_weight, contact_cost_weight=contact_cost_weight)

        #self._ctrl_cost_weight = ctrl_cost_weight
        #self._contact_cost_weight = contact_cost_weight

        #self._healthy_reward = healthy_reward
        #self._terminate_when_unhealthy = terminate_when_unhealthy
        #self._healthy_z_range = healthy_z_range

        #self._contact_force_range = contact_force_range

        #self._reset_noise_scale = reset_noise_scale

        #self._exclude_current_positions_from_observation = (
         #   exclude_current_positions_from_observation)
        self.ctrl_cost_weight = self._ctrl_cost_weight
        self.contact_cost_weight = self._contact_cost_weight
        self.hf_smoothness = hf_smoothness
        self.hf_bump_scale = 2.

        #mujoco_env.MujocoEnv.__init__(self, , 5)

        # Otherwise when learning from scratch might abort
        # This allows for more collisions.
        self.model.nconmax = 500 
        self.model.njmax = 2000
        
#     def _get_obs(self):
#         position = self.sim.data.qpos.flat.copy()
#         velocity = self.sim.data.qvel.flat.copy()
#         contact_force = self.contact_forces.flat.copy()
# 
#         if self._exclude_current_positions_from_observation:
#             position = position[2:]
# 
#         observations = np.concatenate((position, velocity, contact_force))
# 
#         return observations
#         
    def _get_obs(self):
        """ 
        Observation space for the Hexapod model.
        
        Following observation spaces are used: 
        * position information
        * velocity information
        * passive forces acting on the joints
        * last control signal
    
        
        For measured observations (basically everything starting with a q) ordering is:
            FL: 0, 1, 2
            FR: 3, 4, 5
            ML: 6, 7, 8
            MR: 9, 10, 11
            HL: 12, 13, 14
            HR: 15, 16, 17
            Important: plus offset! The first entries are global coordinates, velocities.
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
        

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)
        self.set_state(qpos, qvel)

        observation = self._get_obs()

        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
