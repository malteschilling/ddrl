from gym.envs.mujoco.ant_v3 import AntEnv
import numpy as np

DEFAULT_CAMERA_CONFIG = {
    'distance': 10.0,
}

class QuAntrupedEnv(AntEnv):

    def __init__(self, ctrl_cost_weight=0.5, contact_cost_weight=5e-4, healthy_reward=0.):
        super().__init__(ctrl_cost_weight=ctrl_cost_weight, contact_cost_weight=0.)
        self.ctrl_cost_weight = self._ctrl_cost_weight
        self.contact_cost_weight = self._contact_cost_weight
  #      self.frame_skip = 1

#    @property
 #   def healthy_reward(self):
  #      return 0.

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
        joint_passive_forces = self.sim.data.qfrc_passive.flat.copy()[6:]

        # Provide actions from last time step (as used in the simulator = clipped)
        last_control = self.sim.data.ctrl.flat.copy()
        
        if self._exclude_current_positions_from_observation:
            position = position[2:]

        observations = np.concatenate((position, velocity, joint_passive_forces, last_control))#, last_control)) #, contact_force))

        return observations
        
    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                 getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)