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
    from dynamics import simple
    from util import closed_loop

    # plotting
    import matplotlib.pyplot as plt

    plt.rcParams["text.usetex"] = False
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["pdf.fonttype"] = 42  # TrueType
    plt.rcParams["ps.fonttype"] = 42
    font = {"size": 15}
    plt.rc("font", **font)
    return closed_loop, hj, jnp, mo, np, plt, simple


@app.cell
def _(hj, np, simple):
    # This is where all of the work of HJR happens, namely computing the value function V

    # specify the time horizon of the problem
    T = 8
    t_dose = 2

    # specify the dynamics we are considering

    model = simple.two_compartment_model(kt=1.0, ke=1.0, t_end = t_dose - T, uMax=1)

    # specify the number of voxels to divide the spatial and temporal axes
    x_voxels = 50
    y_voxels = 50
    t_voxels = 100

    # Specify bounds
    x_min = -0.5
    y_min = -0.5

    x_max = +10.5
    y_max = +10.5

    # Specify therapeutic and toxic thresholds
    l_x = 0.5
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

    g = grid.states[..., 1] - g_y
    return T, g, g_y, grid, l, l_x, model, t_dose, times


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

    clRA = closed_loop.ClosedLoopTrajectorySimple(
        model, grid, times, VRA, initial_state=[0.0] * 2, steps=100
    )
    return (clRA,)


@app.cell
def _(T, clRA, g_y, l_x, plt, t_dose, times):
    def _():
        fig, axs = plt.subplots(3,1, figsize = (7,7))

        ax = axs[0]
        ax.plot(times + T, [clRA.u(t)[0] if t + T < t_dose else 0. for t in times], color = 'black')
        ax.set_ylim(-.1,1.1)
        ax.set_xlim(-.1,8.1)
        ax.set_ylabel(r'$\text{Dosing Rate}$')
    
        ax = axs[1]
        ax.plot(times + T, [clRA.x(t)[0] for t in times], color = 'purple')
        ax.set_ylim(-.1,1.1)
        ax.set_xlim(-.1,8.1)
        ax.set_ylabel(r'$[\text{Drug}]_{\text{Blood}}$')
        ax.axhline(l_x, label = 'Therapeutic', color = 'green', linestyle = '--')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        ax = axs[2]
        plt.plot(times + T, [clRA.x(t)[1] for t in times], color = 'blue')
        ax.set_ylim(-.1,1.1)
        ax.set_xlim(-.1,8.1)
        ax.set_xlabel(r'$t$ (hrs)')
        ax.set_ylabel(r'$[\text{Drug}]_{\text{Kidneys}}$')
        ax.axhline(g_y, label = 'Toxic', color = 'red', linestyle = '--')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.tight_layout()
        plt.savefig("/Users/dylanhirsch/Desktop/simple_example_gone_wrong.pdf", transparent=True)
        plt.show()

    _()
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
def _(T, clRAA, g_y, l_x, plt, t_dose, times):
    def _():
        fig, axs = plt.subplots(3,1, figsize = (7,7))

        ax = axs[0]
        ax.plot(times + T, [clRAA.u(t)[0] if t + T < t_dose else 0. for t in times], color = 'black')
        ax.set_ylim(-.1,1.1)
        ax.set_xlim(-.1,8.1)
        ax.set_ylabel(r'$\text{Dosing Rate}$')
    
        ax = axs[1]
        ax.plot(times + T, [clRAA.x(t)[0] for t in times], color = 'purple')
        ax.set_ylim(-.1,1.1)
        ax.set_xlim(-.1,8.1)
        ax.set_ylabel(r'$[\text{Drug}]_{\text{Blood}}$')
        ax.axhline(l_x, label = 'Therapeutic', color = 'green', linestyle = '--')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        ax = axs[2]
        plt.plot(times + T, [clRAA.x(t)[1] for t in times], color = 'blue')
        ax.set_ylim(-.1,1.1)
        ax.set_xlim(-.1,8.1)
        ax.set_xlabel(r'$t$ (hrs)')
        ax.set_ylabel(r'$[\text{Drug}]_{\text{Kidneys}}$')
        ax.axhline(g_y, label = 'Toxic', color = 'red', linestyle = '--')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.tight_layout()
        plt.savefig("/Users/dylanhirsch/Desktop/simple_example_gone_right.pdf", transparent=True)
        plt.show()

    _()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
