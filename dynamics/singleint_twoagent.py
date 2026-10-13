import jax.numpy as jnp
from hj_reachability import dynamics, sets


class singleint_twoagent(dynamics.ControlAndDisturbanceAffineDynamics):
    def __init__(
        self,
        control_mode="min",
        disturbance_mode="max",
        vmax_u=1.0,
        vmax_d=0.5,
    ):

        self.vmax_u = vmax_u
        self.vmax_d = vmax_d
        control_space = sets.Ball(jnp.zeros(2), self.vmax_u)
        disturbance_space = sets.Ball(jnp.zeros(2), self.vmax_d)

        super().__init__(
            control_mode, disturbance_mode, control_space, disturbance_space
        )

    def open_loop_dynamics(self, state, time):

        return jnp.zeros(4)

    def control_jacobian(self, state, time):

        return jnp.eye(4)[:, :2]

    def disturbance_jacobian(self, state, time):

        return jnp.eye(4)[:, 2:]
