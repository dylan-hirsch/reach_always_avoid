import jax.numpy as jnp
from hj_reachability import dynamics, sets


class three_compartment_model(dynamics.ControlAndDisturbanceAffineDynamics):
    def __init__(
        self,
        S1,
        S2,
        control_mode="min",
        disturbance_mode="max",
        uMax=1.0,
        uMin=0.0,
        dR=0.5,
        dC=1.0,
    ):
        self.uMax = uMax
        self.uMin = uMin
        self.dR = dR
        self.dC = dC
        self.S1 = S1
        self.S2 = S2

        control_space = sets.Box(
            jnp.array([self.uMin, self.uMin]), jnp.array([self.uMax, self.uMax])
        )
        disturbance_space = sets.Ball(jnp.array([self.dC, self.dC, self.dC]), self.dR)

        super().__init__(
            control_mode, disturbance_mode, control_space, disturbance_space
        )

    def open_loop_dynamics(self, state, time):

        x, y, z = state

        f = jnp.array([[0.0], [0.0], [0.0]])

        return f.reshape([3])

    def control_jacobian(self, state, time):

        multiplier = jnp.minimum(
            jnp.maximum(1.0 - (time - self.S1) / (self.S2 - self.S1), 0.0), 1.0
        )

        g = jnp.array([[5 * multiplier, 0.0], [0.0, multiplier], [0.0, 0.0]])

        return g.reshape([3, 2])

    def disturbance_jacobian(self, state, time):

        x, y, z = state

        h = jnp.array(
            [
                [-1.0 * x, 0.0 * y, 0.0 * z],
                [1.0 * x, -0.5 * y, 0.0 * z],
                [0.0 * x, 0.5 * y, -1.0 * z],
            ]
        )

        return h.reshape([3, 3])
