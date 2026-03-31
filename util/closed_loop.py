import math

import numpy as np
from numpy.testing import verbose
import scipy as sp
from tqdm.auto import tqdm


def _progress_range(steps, desc):
    return tqdm(range(steps), total=steps, desc=desc, leave=False)


class ClosedLoopTrajectory:
    """
    Docstring for ClosedLoopTrajectory

    This class represents a closed-loop trajectory for a sample-and-hold controller
    designed from the output of hjpy.
    """

    def __init__(
        self,
        model,
        grid,
        times,
        value_function,
        v_off,
        initial_state,
        target=None,
        thresh=-np.inf,
        steps=100,
        verbose=False,
        **kwargs,
    ):
        """
        Docstring for __init__

        :param self: ClosedLoopTrajctory object
        :param model: dynamics object from hjpy
        :param grid: spatial grid object from hjpy
        :param times: times over which the value function was computed
        :param value_function: value function tensor
        :param initial_state: numpy vector representing the initial state
        :param steps: (int) number of sample and hold steps
        :param **kwags: optional arguments to be passed to scipy.integrate.solve_ivp
        """
        self._model = model
        self._grid = grid
        if times[0] < times[-1]:
            raise ValueError(
                "Time axis should start at the final time and move backwards."
            )
        elif times[-1] < times[0]:
            self._times = times[::-1]
            self._V_on = value_function[::-1, ...]
            self._V_off = v_off[::-1, ...]

        self._V = self._V_on
        self._target = target
        self._thresh = thresh

        self._initial_state = np.array(initial_state)
        self._steps = max(int(steps), 1)
        self._grid_mins = np.array(
            [np.asarray(coords)[0] for coords in self._grid.coordinate_vectors]
        )
        self._grid_maxs = np.array(
            [np.asarray(coords)[-1] for coords in self._grid.coordinate_vectors]
        )

        self._u = None
        self._d = None

        self._us = [None] * self._steps
        self._ds = [None] * self._steps
        self._sols = [None] * self._steps
        
        self.verbose = verbose
        self._solve_ivp(**kwargs)

    def x(self, t):
        return self._sols[self._get_sol_index(t)](t)

    def u(self, t):
        return self._us[self._get_sol_index(t)]

    def d(self, t):
        return self._ds[self._get_sol_index(t)]

    def gradient(self, t):
        return self._gradient(t, self.x(t))

    def value(self, t):
        return self._value(t, self.x(t))

    def _gradient(self, t, state, V=None):
        if V is None:
            V = self._V
        j, k = self._get_time_indexes(t)

        if j == k:
            gradient = self._grid.interpolate(
                self._grid.grad_values(V[j, ...]), state=state
            )

        else:
            gradient_left = self._grid.interpolate(
                self._grid.grad_values(V[j, ...]), state=state
            )
            gradient_right = self._grid.interpolate(
                self._grid.grad_values(V[k, ...]), state=state
            )

            gradient = (
                (t - self._times[j]) * gradient_right
                + (self._times[k] - t) * gradient_left
            ) / (self._times[k] - self._times[j])

        return gradient

    def _value(self, t, state, V=None):
        if V is None:
            V = self._V

        j, k = self._get_time_indexes(t)

        if j == k:
            value = self._grid.interpolate(V[j, ...], state=state)

        else:
            value_left = self._grid.interpolate(V[j, ...], state=state)
            value_right = self._grid.interpolate(V[k, ...], state=state)

            # print(f"t: {t:.2f}, value_left: {value_left:.2f}, value_right: {value_right:.2f}")
            # print(f"t: {t:.2f}, time_left: {self._times[j]:.2f}, time_right: {self._times[k]:.2f}")
            # print(f"t: {t:.2f}, state: {state}")

            value = (
                (t - self._times[j]) * value_right + (self._times[k] - t) * value_left
            ) / (self._times[k] - self._times[j])

            # print(f"t: {t:.2f}, value: {value:.2f}")

        return value

    def _dynamics(self, time, state):
        return self._model.__call__(state, self._u, self._d, time)

    def _get_time_indexes(self, t):
        i = np.searchsorted(self._times, t, side="right") - 1
        if t == self._times[i]:
            return i, i
        else:
            return i, i + 1

    def _get_sol_index(self, t):
        if t >= self._times[-1]:
            return -1
        elif t <= self._times[0]:
            return 0
        else:
            return math.floor(
                (t - self._times[0]) / (self._times[-1] - self._times[0]) * self._steps
            )

    def _check_state_in_bounds(self, t, state, label):
        below = state < self._grid_mins
        above = state > self._grid_maxs
        if np.any(below | above):
            print(
                f"t: {t:.2f}, {label} STATE LEFT GRID BOUNDS! "
                f"state: {state}, mins: {self._grid_mins}, maxs: {self._grid_maxs}, "
                f"below: {below}, above: {above}"
            )

    def _solve_ivp(self, **kwargs):
        state = self._initial_state
        switched = False

        for i in _progress_range(self._steps, type(self).__name__):
            t = (self._times[-1] - self._times[0]) * (i / self._steps) + self._times[0]
            t_plus = (self._times[-1] - self._times[0]) * (
                (i + 1) / self._steps
            ) + self._times[0]
            self._check_state_in_bounds(t, state, "pre-solve")

            if (
                not switched
                and self._target is not None
                and self._grid.interpolate(self._target, state=state) <= self._thresh
            ):
                switched = True
                self._V = self._V_off

            if self.verbose:
                print(f"t: {t:.2f}, value: {self._value(t, state):.2f}, switched: {switched}")

            gradient = self._gradient(t, state)

            if switched:
                self._u = 0 * self._model.optimal_control(state, t, gradient)
            else:
                self._u = self._model.optimal_control(state, t, gradient)
            self._d = self._model.optimal_disturbance(state, t, gradient)

            # check if nan
            if np.isnan(self._value(t, state)):
                print(f"t: {t:.2f}, VALUE IS NAN, BREAKING")
                break

            sol = sp.integrate.solve_ivp(
                self._dynamics, [t, t_plus], state, dense_output=True, **kwargs
            )

            state = sol.sol(t_plus)
            self._check_state_in_bounds(t_plus, state, "post-solve")
            self._sols[i] = sol.sol
            self._us[i] = self._u
            self._ds[i] = self._d

            self._V = self._V_on


class ClosedLoopTrajectoryRAA:
    """
    Docstring for ClosedLoopTrajectory

    This class represents a closed-loop trajectory for a sample-and-hold controller
    designed from the output of hjpy.
    """

    def __init__(
        self,
        model,
        grid,
        times,
        VRAA,
        VA,
        target,
        initial_state,
        steps=100,
        theta=1.0,
        **kwargs,
    ):
        """
        Docstring for __init__

        :param self: ClosedLoopTrajctory object
        :param model: dynamics object from hjpy
        :param grid: spatial grid object from hjpy
        :param times: times over which the value function was computed
        :param value_function: value function tensor
        :param initial_state: numpy vector representing the initial state
        :param steps: (int) number of sample and hold steps
        :param **kwags: optional arguments to be passed to scipy.integrate.solve_ivp
        """
        self._model = model
        self._grid = grid
        if times[0] < times[-1]:
            raise ValueError(
                "Time axis should start at the final time and move backwards."
            )
        elif times[-1] < times[0]:
            self._times = times[::-1]
            self._VRAA = VRAA[::-1, ...]
            self._VA = VA[::-1, ...]
            self._target = target
        self._V = self._VRAA

        self._initial_state = np.array(initial_state)
        self._steps = max(int(steps), 1)
        self._grid_mins = np.array(
            [np.asarray(coords)[0] for coords in self._grid.coordinate_vectors]
        )
        self._grid_maxs = np.array(
            [np.asarray(coords)[-1] for coords in self._grid.coordinate_vectors]
        )

        self._u = None
        self._d = None

        self._us = [None] * self._steps
        self._ds = [None] * self._steps
        self._sols = [None] * self._steps

        self._theta = theta

        self._solve_ivp(**kwargs)

    def x(self, t):
        return self._sols[self._get_sol_index(t)](t)

    def u(self, t):
        return self._us[self._get_sol_index(t)]

    def d(self, t):
        return self._ds[self._get_sol_index(t)]

    def gradient(self, t):
        return self._gradient(t, self.x(t))

    def value(self, t):
        return self._value(t, self.x(t))

    def _gradient(self, t, state, V=None):
        if V is None:
            V = self._V
        j, k = self._get_time_indexes(t)

        if j == k:
            gradient = self._grid.interpolate(
                self._grid.grad_values(V[j, ...]), state=state
            )

        else:
            gradient_left = self._grid.interpolate(
                self._grid.grad_values(V[j, ...]), state=state
            )
            gradient_right = self._grid.interpolate(
                self._grid.grad_values(V[k, ...]), state=state
            )

            gradient = (
                (t - self._times[j]) * gradient_right
                + (self._times[k] - t) * gradient_left
            ) / (self._times[k] - self._times[j])

        return gradient

    def _value(self, t, state, V=None):
        if V is None:
            V = self._V

        j, k = self._get_time_indexes(t)

        if j == k:
            value = self._grid.interpolate(V[j, ...], state=state)

        else:
            value_left = self._grid.interpolate(V[j, ...], state=state)
            value_right = self._grid.interpolate(V[k, ...], state=state)

            value = (
                (t - self._times[j]) * value_right + (self._times[k] - t) * value_left
            ) / (self._times[k] - self._times[j])

        return value

    def _dynamics(self, time, state):
        return self._model.__call__(state, self._u, self._d, time)

    def _get_time_indexes(self, t):
        i = np.searchsorted(self._times, t, side="right") - 1
        if t == self._times[i]:
            return i, i
        else:
            return i, i + 1

    def _get_sol_index(self, t):
        if t >= self._times[-1]:
            return -1
        elif t <= self._times[0]:
            return 0
        else:
            return math.floor(
                (t - self._times[0]) / (self._times[-1] - self._times[0]) * self._steps
            )

    def _check_state_in_bounds(self, t, state, label):
        below = state < self._grid_mins
        above = state > self._grid_maxs
        if np.any(below | above):
            print(
                f"t: {t:.2f}, {label} state left grid bounds, "
                f"state: {state}, mins: {self._grid_mins}, maxs: {self._grid_maxs}, "
                f"below: {below}, above: {above}"
            )

    def _solve_ivp(self, **kwargs):
        state = self._initial_state
        switched = False
        for i in _progress_range(self._steps, type(self).__name__):
            t = (self._times[-1] - self._times[0]) * (i / self._steps) + self._times[0]
            t_plus = (self._times[-1] - self._times[0]) * (
                (i + 1) / self._steps
            ) + self._times[0]
            self._check_state_in_bounds(t, state, "pre-solve")

            if not switched and self._grid.interpolate(
                self._target, state=state
            ) <= self._theta * self._value(t, state, self._VA):
                switched = True
                self._V = self._VA

            gradient = self._gradient(t, state)

            self._u = self._model.optimal_control(state, t, gradient)
            self._d = self._model.optimal_disturbance(state, t, gradient)

            sol = sp.integrate.solve_ivp(
                self._dynamics, [t, t_plus], state, dense_output=True, **kwargs
            )

            state = sol.sol(t_plus)
            self._check_state_in_bounds(t_plus, state, "post-solve")
            self._sols[i] = sol.sol
            self._us[i] = self._u
            self._ds[i] = self._d
        self._V = self._VRAA

class ClosedLoopTrajectoryRR:
    """
    Docstring for ClosedLoopTrajectory

    This class represents a closed-loop trajectory for a sample-and-hold controller
    designed from the output of hjpy.
    """

    def __init__(
        self,
        model,
        grid,
        times,
        VRR,
        VR1,
        VR2,
        target_1,
        target_2,
        initial_state,
        steps=100,
        theta=1.0,
        **kwargs,
    ):
        """
        Docstring for __init__

        :param self: ClosedLoopTrajectoryRR object
        :param model: dynamics object from hjpy
        :param grid: spatial grid object from hjpy
        :param times: times over which the value function was computed
        :param value_function: value function tensor
        :param initial_state: numpy vector representing the initial state
        :param steps: (int) number of sample and hold steps
        :param **kwags: optional arguments to be passed to scipy.integrate.solve_ivp
        """
        self._model = model
        self._grid = grid
        if times[0] < times[-1]:
            raise ValueError(
                "Time axis should start at the final time and move backwards."
            )
        elif times[-1] < times[0]:
            self._times = times[::-1]
            self._VRR = VRR[::-1, ...]
            self._VR1 = VR1[::-1, ...]
            self._VR2 = VR2[::-1, ...]
            self._target_1 = target_1
            self._target_2 = target_2

        self._V = self._VRR

        self._initial_state = np.array(initial_state)
        self._steps = max(int(steps), 1)
        self._grid_mins = np.array(
            [np.asarray(coords)[0] for coords in self._grid.coordinate_vectors]
        )
        self._grid_maxs = np.array(
            [np.asarray(coords)[-1] for coords in self._grid.coordinate_vectors]
        )

        self._u = None
        self._d = None

        self._us = [None] * self._steps
        self._ds = [None] * self._steps
        self._sols = [None] * self._steps

        self._theta = theta

        self._solve_ivp(**kwargs)

    def x(self, t):
        return self._sols[self._get_sol_index(t)](t)

    def u(self, t):
        return self._us[self._get_sol_index(t)]

    def d(self, t):
        return self._ds[self._get_sol_index(t)]

    def gradient(self, t):
        return self._gradient(t, self.x(t))

    def value(self, t):
        return self._value(t, self.x(t))

    def _gradient(self, t, state, V=None):
        if V is None:
            V = self._V
        j, k = self._get_time_indexes(t)

        if j == k:
            gradient = self._grid.interpolate(
                self._grid.grad_values(V[j, ...]), state=state
            )

        else:
            gradient_left = self._grid.interpolate(
                self._grid.grad_values(V[j, ...]), state=state
            )
            gradient_right = self._grid.interpolate(
                self._grid.grad_values(V[k, ...]), state=state
            )

            gradient = (
                (t - self._times[j]) * gradient_right
                + (self._times[k] - t) * gradient_left
            ) / (self._times[k] - self._times[j])

        return gradient

    def _value(self, t, state, V=None):
        if V is None:
            V = self._V

        j, k = self._get_time_indexes(t)

        if j == k:
            value = self._grid.interpolate(V[j, ...], state=state)

        else:
            value_left = self._grid.interpolate(V[j, ...], state=state)
            value_right = self._grid.interpolate(V[k, ...], state=state)

            value = (
                (t - self._times[j]) * value_right + (self._times[k] - t) * value_left
            ) / (self._times[k] - self._times[j])

        return value

    def _dynamics(self, time, state):
        return self._model.__call__(state, self._u, self._d, time)

    def _get_time_indexes(self, t):
        i = np.searchsorted(self._times, t, side="right") - 1
        if t == self._times[i]:
            return i, i
        else:
            return i, i + 1

    def _get_sol_index(self, t):
        if t >= self._times[-1]:
            return -1
        elif t <= self._times[0]:
            return 0
        else:
            return math.floor(
                (t - self._times[0]) / (self._times[-1] - self._times[0]) * self._steps
            )

    def _check_state_in_bounds(self, t, state, label):
        below = state < self._grid_mins
        above = state > self._grid_maxs
        if np.any(below | above):
            print(
                f"t: {t:.2f}, {label} state left grid bounds, "
                f"state: {state}, mins: {self._grid_mins}, maxs: {self._grid_maxs}, "
                f"below: {below}, above: {above}"
            )

    def _solve_ivp(self, **kwargs):
        state = self._initial_state
        switched = False
        for i in _progress_range(self._steps, type(self).__name__):
            t = (self._times[-1] - self._times[0]) * (i / self._steps) + self._times[0]
            t_plus = (self._times[-1] - self._times[0]) * (
                (i + 1) / self._steps
            ) + self._times[0]
            self._check_state_in_bounds(t, state, "pre-solve")

            ## SWITCHING CONDITIONS
            if not switched and self._grid.interpolate(
                self._target_1, # FIXME?
                state=state
            ) <= self._theta * self._value(t, state, self._VR2):
                switched = True
                self._V = self._VR2

            elif not switched and self._grid.interpolate(
                self._target_2, # FIXME?
                state=state
            ) <= self._theta * self._value(t, state, self._VR1):
                switched = True
                self._V = self._VR1

            gradient = self._gradient(t, state)

            self._u = self._model.optimal_control(state, t, gradient)
            self._d = self._model.optimal_disturbance(state, t, gradient)

            sol = sp.integrate.solve_ivp(
                self._dynamics, [t, t_plus], state, dense_output=True, **kwargs
            )

            state = sol.sol(t_plus)
            self._check_state_in_bounds(t_plus, state, "post-solve")
            self._sols[i] = sol.sol
            self._us[i] = self._u
            self._ds[i] = self._d
        self._V = self._VRR
