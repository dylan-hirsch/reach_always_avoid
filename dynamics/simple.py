import jax.numpy as jnp
from hj_reachability import dynamics, sets


class two_compartment_model(dynamics.ControlAndDisturbanceAffineDynamics):
    def __init__(
        self,
        kt,
        ke,
        t_end,
        control_mode="min",
        disturbance_mode="max",
        uMax=1.0,
        uMin=0.0,
        dR=0.0,
        dC=0.0,
    ):
        self.uMax = uMax
        self.uMin = uMin
        self.dR = dR
        self.dC = dC
        self.kt = kt
        self.ke = ke
        self.t_end = t_end

        control_space = sets.Box(jnp.array([self.uMin]), jnp.array([self.uMax]))
        disturbance_space = sets.Ball(jnp.array([self.dC]), self.dR)

        super().__init__(
            control_mode, disturbance_mode, control_space, disturbance_space
        )

    def open_loop_dynamics(self, state, time):

        x, y = state

        f = jnp.array([[-self.kt * x], [self.kt * x - self.ke * y]])

        return f.reshape([2])

    def control_jacobian(self, state, time):

        g = jnp.array([[jnp.where(time < self.t_end, 1.0, 0.0)], [0.0]])

        return g.reshape([2, 1])

    def disturbance_jacobian(self, state, time):

        x, y = state

        h = jnp.array(
            [
                [0.0],
                [0.0],
            ]
        )

        return h.reshape([2, 1])
