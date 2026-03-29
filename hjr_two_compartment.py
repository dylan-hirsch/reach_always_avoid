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
    font = {"size": 12}
    plt.rc("font", **font)
    return closed_loop, hj, jnp, mo, np, plt, two_compartment_model


@app.cell
def _(hj, jnp, np, two_compartment_model):
    # This is where all of the work of HJR happens, namely computing the value function V

    # specify the time horizon of the problem
    T = 14

    # specify the weening period
    W_start = 6  # start weening
    W_end = 8  # stop dosing

    # specify the dynamics we are considering

    model = two_compartment_model.two_compartment_model(
        S1=W_start - T, S2=W_end - T
    )

    # specify the number of voxels to divide the spatial and temporal axes
    x_voxels = 250
    y_voxels = 250
    t_voxels = 500

    # Specify bounds
    x_min = -0.5
    y_min = -0.5

    x_max = +10.5
    y_max = +10.5

    # Specify therapeutic and toxic thresholds
    l_x = 0.5
    g_x = 2.0
    g_y = 0.5

    # discretize state-space and the time to solve the HJ Partial Differential Equation
    # don't change
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box([x_min, y_min], [x_max, y_max]),
        [x_voxels + 1, y_voxels + 1],
    )

    times = np.linspace(0.0, -T, t_voxels + 1)

    # specify the goal
    l = l_x - grid.states[..., 0]

    g = jnp.maximum(grid.states[..., 0] - g_x, grid.states[..., 1] - g_y)
    return T, W_end, W_start, g, g_x, g_y, grid, l, l_x, model, times


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## First, solve a normal RA problem
    """)
    return


@app.cell
def _(g, grid, hj, jnp, l, model, np, times):
    def _():

        # specify the reach-avoid problem
        def value_postprocessor(t, v, l, g):
            return jnp.maximum(jnp.minimum(v, l), g)

        # specify the accuracy with which to solve the HJ Partial Differential Equation
        solver_settings = hj.SolverSettings.with_accuracy(
            "very_high",
            value_postprocessor=lambda t, v: value_postprocessor(t, v, l, g),
        )

        # solve for the value function
        V = hj.solve(solver_settings, model, grid, times, np.maximum(l, g))

        return V


    VRA = _()
    return (VRA,)


@app.cell
def _(VRA, closed_loop, grid, model, times):
    # Form the closed-loop trajectory

    clRA = closed_loop.ClosedLoopTrajectory(
        model, grid, times, VRA, initial_state=[0.0] * 2, steps=100
    )
    return (clRA,)


@app.cell
def _(T, W_end, W_start, clRA, g_x, g_y, l_x, np, plt):
    # This is all just plotting


    def _(cl):

        fig, axs = plt.subplots(2, 2, figsize=(8.5, 8.5))

        ts = np.linspace(-T, 0, 1000)

        ax = axs[0, 0]
        labels = [
            r"$[\text{Drug}]_{\text{Blood}}$",
            r"$[\text{Drug}]_{\text{Kidneys}}$",
            r"$\theta$",
        ]
        ax.plot(ts + T, [cl.x(t)[0] for t in ts], label=labels[0], color="red")
        ax.set_ylim([-0.1, 2.1])
        ax.set_xlim([0, T + 0.1])
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$\mathbf{x}_1(t)$")
        ax.set_title("Blood Compartment")

        ax.axhline(y=g_x, linestyle="--", color="black", label="Toxic Thresh.")
        ax.axhline(
            y=l_x, linestyle="--", color="green", label="Therapeutic Thresh."
        )

        ax.legend()

        ax = axs[1, 0]
        ax.plot(ts + T, [cl.x(t)[1] for t in ts], label=labels[1], color="Brown")
        ax.set_ylim([-0.1, 2.1])
        ax.set_xlim([0, T + 0.1])
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$\mathbf{x}_2(t)$")
        ax.set_title("Kidney Compartment")

        ax.axhline(y=g_y, linestyle="--", color="black", label="Toxic Thresh.")
        ax.legend()

        ax = axs[0, 1]
        ax.plot(
            ts + T,
            [
                cl.u(t) * min(1, max(0, 1 - (t + T - W_start) / (W_end - W_start)))
                for t in ts
            ],
            label=r"$\text{Dosing Rate}$",
        )
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$\mathbf{u}^*(t)$")
        ax.set_xlim([0, T + 0.1])
        ax.set_title("Optimal Dosing Rate")
        ax.legend()

        ax = axs[1, 1]
        ax.plot(ts + T, [cl.d(t) for t in ts], label=r"$\text{Clearance Rate}$")
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$\mathbf{d}^*(t)$")
        ax.set_xlim([0, T + 0.1])
        ax.set_title("Worst Case Clearance")
        ax.legend()

        plt.tight_layout()

        plt.savefig("/Users/dylanhirsch/Desktop/RA.png")
        plt.show()


    _(clRA)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next, solve just the A problem
    """)
    return


@app.cell
def _(g, grid, hj, jnp, l, model, times):
    def _():

        # specify the reach-avoid problem
        def value_postprocessor(t, v, l, g):
            return jnp.maximum(v, g)

        # specify the accuracy with which to solve the HJ Partial Differential Equation
        solver_settings = hj.SolverSettings.with_accuracy(
            "very_high",
            value_postprocessor=lambda t, v: value_postprocessor(t, v, l, g),
        )

        # solve for the value function
        V = hj.solve(solver_settings, model, grid, times, g)

        return V


    VA = _()
    return (VA,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Finally, solve the RAA
    """)
    return


@app.cell
def _(VA, g, grid, hj, jnp, l, model, np, times):
    def _(VA):

        # specify the reach-avoid problem
        def value_postprocessor(t, v, l, g, times, VA):
            i = jnp.argmin(jnp.abs(t - times))
            return jnp.maximum(jnp.minimum(v, jnp.maximum(l, VA[i, ...])), g)

        # specify the accuracy with which to solve the HJ Partial Differential Equation
        solver_settings = hj.SolverSettings.with_accuracy(
            "very_high",
            value_postprocessor=lambda t, v: value_postprocessor(
                t, v, l, g, times, VA
            ),
        )

        # solve for the value function
        V = hj.solve(
            solver_settings,
            model,
            grid,
            times,
            np.maximum.reduce([l, VA[0, ...], g]),
        )

        return V


    VRAA = _(VA)
    return (VRAA,)


@app.cell
def _(VA, VRAA, closed_loop, grid, l, model, times):
    # Form the closed-loop trajectory

    clRAA = closed_loop.ClosedLoopTrajectoryRAA(
        model, grid, times, VRAA, VA, l, initial_state=[0.0] * 2, steps=100
    )
    return (clRAA,)


@app.cell
def _(T, W_end, W_start, clRAA, g_x, g_y, l_x, np, plt):
    # This is all just plotting


    def _(cl):

        fig, axs = plt.subplots(2, 2, figsize=(8.5, 8.5))

        ts = np.linspace(-T, 0, 1000)

        ax = axs[0, 0]
        labels = [
            r"$[\text{Drug}]_{\text{Blood}}$",
            r"$[\text{Drug}]_{\text{Kidneys}}$",
            r"$\theta$",
        ]
        ax.plot(ts + T, [cl.x(t)[0] for t in ts], label=labels[0], color="red")
        ax.set_ylim([-0.1, 2.1])
        ax.set_xlim([0, T + 0.1])
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$\mathbf{x}_1(t)$")
        ax.set_title("Blood Compartment")

        ax.axhline(y=g_x, linestyle="--", color="black", label="Toxic Thresh.")
        ax.axhline(
            y=l_x, linestyle="--", color="green", label="Therapeutic Thresh."
        )

        ax.legend()

        ax = axs[1, 0]
        ax.plot(ts + T, [cl.x(t)[1] for t in ts], label=labels[1], color="Brown")
        ax.set_ylim([-0.1, 2.1])
        ax.set_xlim([0, T + 0.1])
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$\mathbf{x}_2(t)$")
        ax.set_title("Kidney Compartment")

        ax.axhline(y=g_y, linestyle="--", color="black", label="Toxic Thresh.")
        ax.legend()

        ax = axs[0, 1]
        ax.plot(
            ts + T,
            [
                cl.u(t) * min(1, max(0, 1 - (t + T - W_start) / (W_end - W_start)))
                for t in ts
            ],
            label=r"$\text{Dosing Rate}$",
        )
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$\mathbf{u}^*(t)$")
        ax.set_xlim([0, T + 0.1])
        ax.set_title("Optimal Dosing Rate")
        ax.legend()

        ax = axs[1, 1]
        ax.plot(ts + T, [cl.d(t) for t in ts], label=r"$\text{Clearance Rate}$")
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$\mathbf{d}^*(t)$")
        ax.set_xlim([0, T + 0.1])
        ax.set_title("Worst Case Clearance")
        ax.legend()

        plt.tight_layout()

        plt.savefig("/Users/dylanhirsch/Desktop/RAA.png")
        plt.show()


    _(clRAA)
    return


if __name__ == "__main__":
    app.run()
