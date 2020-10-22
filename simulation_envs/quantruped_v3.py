from gym.envs.mujoco.ant_v3 import AntEnv
import numpy as np

DEFAULT_CAMERA_CONFIG = {
    'distance': 10.0,
}

class QuAntrupedEnv(AntEnv):

    @property
    def healthy_reward(self):
        return 0.
        
    @property
    def contact_cost(self):
        return 0.

    def _get_obs(self):
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