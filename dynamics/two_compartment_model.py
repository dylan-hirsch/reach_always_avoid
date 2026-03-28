import jax.numpy as jnp
from hj_reachability import dynamics, sets


class two_compartment_model(dynamics.ControlAndDisturbanceAffineDynamics):
    def __init__(
        self,
        control_mode="min",
        disturbance_mode="max",
        uMax=1.0,
        uMin=0.0,
        dMax=1.2,
        dMin=0.8,
    ):
        self.uMax = uMax
        self.uMin = uMin
        self.dMax = dMax
        self.dMin = dMin

        control_space = sets.Box(jnp.array([self.uMin]), jnp.array([self.uMax]))
        disturbance_space = sets.Box(jnp.array([self.dMin]), jnp.array([self.dMax]))

        super().__init__(
            control_mode, disturbance_mode, control_space, disturbance_space
        )

    def open_loop_dynamics(self, state, time):

        x, y = state

        f = jnp.array([[-0.5 * x], [0.5 * x]])

        return f.reshape([2])

    def control_jacobian(self, state, time):

        g = jnp.array([[1.0], [0.0]])

        return g.reshape([2, 1])

    def disturbance_jacobian(self, state, time):

        _, y = state

        h = jnp.array([[0.0], [-1.0 * y]])

        return h.reshape([2, 1])
