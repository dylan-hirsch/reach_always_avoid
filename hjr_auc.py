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
    from dynamics import auc
    from util import closed_loop

    # plotting
    import matplotlib.pyplot as plt

    plt.rcParams["text.usetex"] = False
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["pdf.fonttype"] = 42  # TrueType
    plt.rcParams["ps.fonttype"] = 42
    font = {"size": 12}
    plt.rc("font", **font)
    return auc, closed_loop, hj, jnp, mo, np, plt


@app.cell
def _(auc, hj, np):
    # This is where all of the work of HJR happens, namely computing the value function V

    # specify the time horizon of the problem
    T = 14

    # specify the weening period
    W_start = 6  # start weening
    W_end = 8  # stop dosing

    # specify the dynamics we are considering

    model = auc.two_compartment_auc_model(S1=W_start - T, S2=W_end - T)

    model_off = auc.two_compartment_auc_model(
        S1=W_start - T, S2=W_end - T, uMax=0.0, uMin=0.0
    )

    # specify the number of voxels to divide the spatial and temporal axes
    x_voxels = 50
    y_voxels = 50
    z_voxels = 50
    t_voxels = 100

    # Specify bounds
    x_min = -0.5
    y_min = -0.5
    z_min = -0.5

    x_max = +10.5
    y_max = +10.5
    z_max = +10.5

    # Specify therapeutic and toxic thresholds
    l_x = 1.0
    g_z = 1.0

    # discretize state-space and the time to solve the HJ Partial Differential Equation
    # don't change
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box([x_min, y_min, z_min], [x_max, y_max, z_max]),
        [x_voxels + 1, y_voxels + 1, z_voxels + 1],
    )

    times = np.linspace(0.0, -T, t_voxels + 1)

    # specify the goal
    l = l_x - grid.states[..., 2]

    g = grid.states[..., 1] - g_z
    return T, W_end, W_start, g, g_z, grid, l, l_x, model, model_off, times


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## First, solve a normal RA problem
    """)
    return


@app.cell
def _(g, grid, hj, jnp, l, model, model_off, np, times):
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


    def voff():

        # specify the reach-avoid problem
        def value_postprocessor(t, v, g):
            return jnp.maximum(v, g)

        # specify the accuracy with which to solve the HJ Partial Differential Equation
        solver_settings = hj.SolverSettings.with_accuracy(
            "very_high",
            value_postprocessor=lambda t, v: value_postprocessor(t, v, g),
        )

        # solve for the value function
        V = hj.solve(solver_settings, model_off, grid, times, g)

        return V


    VoffA = voff()
    return VRA, VoffA


@app.cell
def _(VRA, VoffA, closed_loop, grid, l, model, times):
    # Form the closed-loop trajectory

    clRA = closed_loop.ClosedLoopTrajectory(
        model, grid, times, VRA, VoffA, initial_state=[0.0] * 3, steps=100
    )

    clRA_cons = closed_loop.ClosedLoopTrajectory(
        model,
        grid,
        times,
        VRA,
        VoffA,
        initial_state=[0.0] * 3,
        target=l,
        thresh=0,
        steps=100,
    )
    return clRA, clRA_cons


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
        model, grid, times, VRAA, VA, l, initial_state=[0.0] * 3, steps=100
    )
    return (clRAA,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## We now build the main figures for the paper
    """)
    return


@app.cell
def _(T, W_end, W_start, clRA, clRAA, clRA_cons, g_z, l_x, np, plt):
    def _():

        time_var = r"$\tau$"

        # ── Helpers ────────────────────────────────────────────────────────────────────

        def omega(t):
            S1, S2 = W_start - T, W_end - T
            return np.clip(1.0 - (t - S1) / (S2 - S1), 0.0, 1.0)

        def style_ax(ax, ylabel, title, xlim=(0, T + 0.1), ylim=None):
            ax.set_xlabel(time_var, fontsize=LABEL_FONT)
            ax.set_ylabel(ylabel, fontsize=LABEL_FONT)
            ax.set_title(title, fontsize=LABEL_FONT)
            ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)

        # ── Style & constants ──────────────────────────────────────────────────────────

        plt.rcParams.update(
            {
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.grid": True,
                "grid.alpha": 0.25,
                "grid.linestyle": "--",
                "font.size": 11,
            }
        )

        LEGEND_FONT = 11
        LABEL_FONT = 14
        LW = 2.5

        # (full-course color, partial-course color) per compartment
        COLORS = {
            "blood": ("black", "black", "lightblue"),
            "kidneys": ("black", "black", "lightblue"),
            "auc": ("black", "black", "lightblue"),
        }
        GREY = "#888888"

        # ── Precompute trajectories ────────────────────────────────────────────────────

        ts = np.linspace(-T, 0, 1000)
        tt = ts + T

        x_full = np.array([clRA.x(t) for t in ts])
        x_cons = np.array([clRA_cons.x(t) for t in ts])
        x_raa = np.array([clRAA.x(t) for t in ts])
        u_full = np.array([clRA.u(t) for t in ts])
        u_cons = np.array([clRA_cons.u(t) for t in ts])
        u_raa = np.array([clRAA.u(t) for t in ts])
        d_full = np.array([clRA.d(t) for t in ts])
        d_cons = np.array([clRA_cons.d(t) for t in ts])
        d_raa = np.array([clRAA.d(t) for t in ts])
        om = np.array([omega(t) for t in ts])

        # ── Figure ─────────────────────────────────────────────────────────────────────

        fig, axs = plt.subplots(3, 2, figsize=(10, 9), constrained_layout=True)

        # Left column – compartment concentrations
        compartments = [
            (0, r"$\mathbf{x}_1(\tau)$", "Drug Concentration (Blood)", "blood", 2),
            (
                1,
                r"$\mathbf{x}_2(\tau)$",
                "Drug Concentration (Kidneys)",
                "kidneys",
                2,
            ),
            (2, r"$\mathbf{x}_3(\tau)$", "AUC", "auc", 6),
        ]
        for row, (i, ylabel, title, ckey, ylim) in enumerate(compartments):
            ax = axs[row, 0]
            cf, cp, cr = COLORS[ckey]
            ax.plot(
                tt, x_full[:, i], color=cf, linewidth=LW, label="RA", linestyle=":"
            )
            ax.plot(
                tt,
                x_cons[:, i],
                color=cp,
                linewidth=LW,
                linestyle="-.",
                label="RA (early stop)",
            )
            ax.plot(
                tt,
                x_raa[:, i],
                color=cr,
                linewidth=LW,
                linestyle="-",
                label="RAA",
            )
            style_ax(ax, ylabel, title, ylim=(-0.1, ylim + 0.1))

        axs[0, 0].legend(ncol=1, fontsize=LEGEND_FONT, loc="upper left")
        axs[1, 0].axhline(
            y=g_z,
            linestyle="--",
            color="red",
            linewidth=LW,
            label=r"Toxic Thresh. $\theta_{\mathrm{toxic}}$",
        )
        axs[1, 0].legend(fontsize=LEGEND_FONT)
        axs[2, 0].axhline(
            y=l_x,
            linestyle="--",
            color="#27AE60",
            linewidth=LW,
            label=r"Therap. Thresh. $\theta_{\mathrm{ther}}$",
        )
        axs[2, 0].legend(fontsize=LEGEND_FONT)

        # Right column (rows 0–1) – dosing rates
        dosing = [
            (
                0,
                1,
                r"$\mathbf{M}(\tau)\,\mathbf{u}^*(\tau)$",
                "Dosing Rate (IV)",
                "blood",
            ),
        ]
        for row, (i, scale, ylabel, title, ckey) in enumerate(dosing):
            ax = axs[row, 1]
            cf, cp, cr = COLORS[ckey]
            ax.plot(
                tt,
                scale * om * u_full[:, i],
                color=cf,
                linewidth=LW,
                linestyle=":",
            )
            ax.plot(
                tt,
                scale * om * u_cons[:, i],
                color=cp,
                linewidth=LW,
                linestyle="-.",
            )
            ax.plot(
                tt,
                scale * om * u_raa[:, i],
                color=cr,
                linewidth=LW,
                linestyle="-",
            )
            ax.plot(
                tt,
                scale * om,
                color=GREY,
                linewidth=LW,
                linestyle="--",
                label=r"$\mathbf{M}$",
            )
            style_ax(ax, ylabel, title)
            ax.legend(fontsize=LEGEND_FONT + 3)

        # Right column (row 2) – worst-case unmodeled dynamics
        for i, (ckey, ylabel, title) in enumerate(
            zip(
                ["blood", "kidneys"],
                [r"$\mathbf{d}_1(\tau)$", r"$\mathbf{d}_2(\tau)$"],
                [
                    "Worst-Case Transport Disturbance",
                    "Worst-Case Elimination Disturbance",
                ],
            )
        ):
            ax = axs[i + 1, 1]
            cf, cp, cr = COLORS[ckey]
            ax.plot(tt, d_full[:, i], color=cf, linewidth=LW, linestyle=":")
            ax.plot(tt, d_cons[:, i], color=cp, linewidth=LW, linestyle="-.")
            ax.plot(tt, d_cons[:, i], color=cr, linewidth=LW, linestyle="-")

            style_ax(ax, ylabel, title, ylim=(-0.1, 2.1))


    _()
    plt.savefig("/Users/dylanhirsch/Desktop/raa.pdf")
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
