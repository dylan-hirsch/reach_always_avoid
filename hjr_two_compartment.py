import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    # fundamental numerical library
    import hj_reachability as hj
    import numpy as np
    import scipy as sp
    import jax.numpy as jnp

    # my dynamics
    from dynamics import two_compartment_model
    from util import closed_loop

    # plotting
    import matplotlib.pyplot as plt

    plt.rcParams["text.usetex"] = False
    plt.rcParams["mathtext.fontset"] = "cm"
    font = {"size": 20}
    plt.rc("font", **font)
    return closed_loop, hj, jnp, np, plt, two_compartment_model


@app.cell
def _(hj, jnp, np, two_compartment_model):
    # This is where all of the work of HJR happens, namely computing the value function V

    # specify the dynamics we are considering
    model = two_compartment_model.two_compartment_model()

    # specify the time horizon of the problem
    T = 10

    # specify the number of voxels to divide the spatial and temporal axes
    x_voxels = 100
    y_voxels = 100
    theta_voxels = 100
    t_voxels = 200

    # Specify bounds
    x_min = -10
    y_min = -10

    x_max = +10
    y_max = +10

    # discretize state-space and the time to solve the HJ Partial Differential Equation
    # don't change
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box([x_min, y_min], [x_max, y_max]),
        [x_voxels + 1, y_voxels + 1],
    )

    times = np.linspace(0.0, -T, t_voxels + 1)

    # specify the goal
    l = (
        np.sqrt(
            (grid.states[..., 0] - 1.0) ** 2 + (grid.states[..., 1] - 1.0) ** 2
        )
        - 0.5
    )

    g = -(
        np.sqrt(
            (grid.states[..., 0] - 1.0) ** 2 + (grid.states[..., 1] - 0.0) ** 2
        )
        + 0.5
    )


    # specify the reach-avoid problem
    def value_postprocessor(t, v, l, g):
        return jnp.maximum(jnp.minimum(v, l), g)


    # specify the accuracy with which to solve the HJ Partial Differential Equation
    solver_settings = hj.SolverSettings.with_accuracy(
        "very_high",
        value_postprocessor=lambda t, v: value_postprocessor(t, v, l, g),
    )

    # solve for the value function
    V = hj.solve(solver_settings, model, grid, times, l)
    return T, V, grid, l, model, times


@app.cell
def _(V, closed_loop, grid, model, times):
    # Form the closed-loop trajectory

    cl = closed_loop.ClosedLoopTrajectory(
        model, grid, times, V, initial_state=[0.0] * 2, steps=100
    )
    return (cl,)


@app.cell
def _(T, cl, l, np, plt):
    # This is all just plotting

    fig, axs = plt.subplots(1, 4, figsize=(20, 6))

    ts = np.linspace(-T, 0, 1000)

    ax = axs[0]
    labels = [r"$x$", r"$y$", r"$\theta$"]
    for i in range(2):
        ax.plot(ts, [cl.x(t)[i] for t in ts], label=labels[i])
    ax.set_ylim([-10.1, 10.1])
    ax.set_xlim([-T, 0.1])
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\mathbf{x}(t)$")
    ax.legend()
    ax.set_title("State Trajectory")

    ax = axs[1]
    ax.plot(ts, [cl.u(t) for t in ts], label=r"$\omega$")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$u^*(t)$")
    ax.set_xlim([-T, 0.1])
    ax.set_title("Optimal Control")
    ax.legend()

    ax = axs[2]
    ax.plot(ts, [cl.value(t) for t in ts])
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$V(\mathbf{x}(t), t)$")
    ax.set_ylim([np.min(l) - 0.1, 1.0])
    ax.axhline([0.0], color="black", linestyle="--")
    ax.set_xlim([-T, 0.1])
    ax.set_title("Value")

    ax = axs[3]
    ax.plot(
        [cl.x(t)[0] for t in ts],
        [cl.x(t)[1] for t in ts],
        label="trajectory",
    )
    ax.set_ylim([-10.1, 10.1])
    ax.set_xlim([-10.1, 10.1])
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")

    phis = np.linspace(0, 2 * np.pi, 101)
    ax.plot(
        [1 + 0.5 * np.cos(phi) for phi in phis],
        [1 + 0.5 * np.sin(phi) for phi in phis],
        label="target",
    )
    ax.plot(
        [1 + 0.5 * np.cos(phi) for phi in phis],
        [0 + 0.5 * np.sin(phi) for phi in phis],
        label="obstacle",
    )
    ax.scatter([0], [0], color="red", label="start")
    ax.legend()
    ax.set_title("Phase Plane (x-y)")

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
