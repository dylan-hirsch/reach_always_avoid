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
    from dynamics import three_compartment_model
    from util import closed_loop

    # plotting
    import matplotlib.pyplot as plt

    plt.rcParams["text.usetex"] = False
    plt.rcParams["mathtext.fontset"] = "cm"
    font = {"size": 12}
    plt.rc("font", **font)
    return closed_loop, hj, jnp, mo, np, plt, three_compartment_model


@app.cell
def _(hj, np, three_compartment_model):
    # This is where all of the work of HJR happens, namely computing the value function V

    # specify the time horizon of the problem
    T = 14

    # specify the weening period
    W_start = 6  # start weening
    W_end = 8  # stop dosing

    # specify the dynamics we are considering

    model = three_compartment_model.three_compartment_model(
        S1=W_start - T, S2=W_end - T
    )

    model_off = three_compartment_model.three_compartment_model(
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
    l_x = 0.4
    g_z = 0.5

    # discretize state-space and the time to solve the HJ Partial Differential Equation
    # don't change
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box([x_min, y_min, z_min], [x_max, y_max, z_max]),
        [x_voxels + 1, y_voxels + 1, z_voxels + 1],
    )

    times = np.linspace(0.0, -T, t_voxels + 1)

    # specify the goal
    l = l_x - grid.states[..., 1]

    g = grid.states[..., 2] - g_z
    return T, W_end, W_start, g, g_z, grid, l, l_x, model, model_off, times


@app.cell
def _(T, W_end, W_start, g_z, l_x, np, plt):
    linewidth = 2

    def plot(cl):
        # This is all just plotting

        fig, axs = plt.subplots(3, 2, figsize=(8, 12))

        ts = np.linspace(-T, 0, 1000)

        labels = [
            r"$[\text{Drug}]_{\text{Blood}}$",
            r"$[\text{Drug}]_{\text{Kidneys}}$",
            r"$\theta$",
        ]

        # gut compartment
        ax = axs[0, 0]
        ax.plot(ts + T, [cl.x(t)[0] for t in ts], label=labels[0], color="magenta")
        ax.set_ylim([-0.1, 2.1])
        ax.set_xlim([0, T + 0.1])
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$\mathbf{x}_1(t)$")
        ax.set_title("Gut Compartment")

        ax.legend()

        # blood compartment
        ax = axs[1, 0]
        ax.plot(ts + T, [cl.x(t)[1] for t in ts], label=labels[1], color="red")
        ax.set_ylim([-0.1, 2.1])
        ax.set_xlim([0, T + 0.1])
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$\mathbf{x}_2(t)$")
        ax.set_title("Blood Compartment")

        ax.axhline(
            y=l_x, linestyle="--", color="green", label="Therapeutic Thresh.", linewidth = linewidth
        )
        ax.legend()

        # kidney compartment
        ax = axs[2, 0]
        ax.plot(ts + T, [cl.x(t)[2] for t in ts], label=labels[1], color="brown")
        ax.set_ylim([-0.1, 2.1])
        ax.set_xlim([0, T + 0.1])
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$\mathbf{x}_2(t)$")
        ax.set_title("Kidney Compartment")

        ax.axhline(y=g_z, linestyle="--", color="black", label="Toxic Thresh.", linewidth = linewidth)
        ax.legend()

        # Optimal dosing
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
        plt.show()

    return (plot,)


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
def _(VRA, VoffA, closed_loop, grid, l, model, plot, times):
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


    plot(clRA_cons)
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
def _(VA, VRAA, closed_loop, grid, l, model, plot, times):
    # Form the closed-loop trajectory

    clRAA = closed_loop.ClosedLoopTrajectoryRAA(
        model, grid, times, VRAA, VA, l, initial_state=[0.0] * 3, steps=100
    )

    plot(clRAA)
    return (clRAA,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## We now build the main figures for the paper
    """)
    return


@app.cell
def _():
    """
    # Def omega
    def omega(t):
        S1 = W_start - T
        S2 = W_end - T
        return np.minimum(np.maximum(1.0 - (t - S1) / (S2 - S1), 0.0), 1.0)


    # This is all just plotting

    legend_font = 12
    label_font = 15
    linewidth = 3
    markersize = 5

    fig, axs = plt.subplots(3, 2, figsize=(8, 8))

    ts = np.linspace(-T, 0, 1000)

    # gut compartment
    ax = axs[0, 0]
    ax.plot(
        ts + T, [clRA.x(t)[0] for t in ts], label="Full\ncourse", color="magenta", linewidth = linewidth
    )
    ax.plot(
        ts + T,
        [clRA_cons.x(t)[0] for t in ts],
        label="Partial\ncourse",
        color="pink",
        linestyle=":",
        linewidth = linewidth
    )
    ax.set_ylim([-0.1, 4.1])
    ax.set_xlim([0, T + 0.1])
    ax.set_xlabel(r"$t$", fontsize=label_font)
    ax.set_ylabel(r"$\mathbf{x}_1(t)$", fontsize=label_font)
    ax.set_title(r"Drug Concentration (Gut)")

    ax.legend(ncol=1, fontsize=legend_font, loc="upper left")

    # blood compartment
    ax = axs[1, 0]
    ax.plot(ts + T, [clRA.x(t)[1] for t in ts], color="red", linewidth = linewidth)
    ax.plot(ts + T, [clRA_cons.x(t)[1] for t in ts], color="orange", linestyle=":", linewidth = linewidth)
    ax.set_ylim([-0.1, 4.1])
    ax.set_xlim([0, T + 0.1])
    ax.set_xlabel(r"$t$", fontsize=label_font)
    ax.set_ylabel(r"$\mathbf{x}_2(t)$", fontsize=label_font)
    ax.set_title(r"Drug Concentration (Blood)")

    ax.axhline(y=l_x, linestyle="--", color="green", label="Therap.\nThresh.", linewidth = linewidth)
    ax.legend(fontsize=legend_font)

    # kidney compartment
    ax = axs[2, 0]
    ax.plot(ts + T, [clRA.x(t)[2] for t in ts], color="brown", linewidth = linewidth)
    ax.plot(ts + T, [clRA_cons.x(t)[2] for t in ts], color="tan", linestyle=":", linewidth = linewidth)
    ax.set_ylim([-0.1, 4.1])
    ax.set_xlim([0, T + 0.1])
    ax.set_xlabel(r"$t$", fontsize=label_font)
    ax.set_ylabel(r"$\mathbf{x}_3(t)$", fontsize=label_font)
    ax.set_title(r"Drug Concentration (Kidneys)")

    ax.axhline(y=g_z, linestyle="--", color="black", label="Toxic\nThresh.", linewidth = linewidth)
    ax.legend(fontsize=legend_font)

    # Optimal dosing gut
    ax = axs[0, 1]
    ax.plot(
        ts + T,
        [5 * omega(t) * clRA.u(t)[0] for t in ts],
        color="magenta",
        linewidth = linewidth
    )
    ax.plot(
        ts + T,
        [5 * omega(t) * clRA_cons.u(t)[0] for t in ts],
        color="pink",
        linestyle=":",
        linewidth = linewidth
    )
    ax.set_xlabel(r"$t$", fontsize=label_font)
    ax.set_ylabel(r"$\omega_1(t)\mathbf{u}_1^*(t)$", fontsize=label_font)
    ax.set_xlim([0, T + 0.1])
    ax.set_title("Dosing Rate (Oral)")
    ax.plot(
        ts + T,
        [5 * omega(t) for t in ts],
        linestyle="--",
        color="grey",
        label=r"$\omega_1$",
        linewidth = linewidth
    )
    ax.legend(fontsize=legend_font + 3)

    # Optimal dosing blood
    ax = axs[1, 1]
    ax.plot(
        ts + T,
        [omega(t) * clRA.u(t)[1] for t in ts],
        color="red",
        linewidth = linewidth
    )
    ax.plot(
        ts + T,
        [omega(t) * clRA_cons.u(t)[1] for t in ts],
        color="orange",
        linestyle=":",
        linewidth = linewidth
    )
    ax.set_xlabel(r"$t$", fontsize=label_font)
    ax.set_ylabel(r"$\omega_2(t)\mathbf{u}_2^*(t)$", fontsize=label_font)
    ax.set_xlim([0, T + 0.1])
    ax.set_title("Dosing Rate (IV)")
    ax.plot(
        ts + T,
        [omega(t) for t in ts],
        linestyle="--",
        color="grey",
        label=r"$\omega_2$",
        linewidth = linewidth
    )
    ax.legend(fontsize=legend_font + 3)


    # Optimal clearance
    ax = axs[2, 1]
    ax.plot(
        ts + T,
        [clRA.d(t)[0] for t in ts],
        color="magenta",
        label=r"$\mathbf{d}_1$",
        linewidth = linewidth
    )
    ax.plot(
        ts + T,
        [clRA_cons.d(t)[0] for t in ts],
        linestyle=":",
        color="pink",
        label=r"$\mathbf{d}_1$",
        linewidth = linewidth
    )
    ax.plot(
        ts + T,
        [clRA.d(t)[1] for t in ts],
        color="red",
        label=r"$\mathbf{d}_2$",
        linewidth = linewidth
    )
    ax.plot(
        ts + T,
        [clRA_cons.d(t)[1] for t in ts],
        linestyle=":",
        color="orange",
        label=r"$\mathbf{d}_2$",
        linewidth = linewidth
    )
    ax.plot(
        ts + T,
        [clRA.d(t)[2] for t in ts],
        color="brown",
        label=r"$\mathbf{d}_3$",
        linewidth = linewidth
    )
    ax.plot(
        ts + T,
        [clRA_cons.d(t)[2] for t in ts],
        linestyle=":",
        color="tan",
        label=r"$\mathbf{d}_3$",
        linewidth = linewidth
    )


    ax.set_xlabel(r"$t$", fontsize=label_font)
    ax.set_ylabel(r"$\mathbf{d}^*(t)$", fontsize=label_font)
    ax.set_xlim([0, T + 0.1])
    ax.set_ylim([-0.1, 2.1])
    ax.set_title("Worst Case Unmodeled Dynamics")
    ax.legend(fontsize=legend_font, loc="upper left", facecolor='white', framealpha=1, frameon=True)

    plt.tight_layout()

    plt.savefig("raa.pdf")
    plt.show()
    """
    return


@app.cell
def _(T, W_end, W_start, clRA, clRA_cons, g_z, l_x, np, plt):
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
    
        plt.rcParams.update({
            "axes.spines.top":    False,
            "axes.spines.right":  False,
            "axes.grid":          True,
            "grid.alpha":         0.25,
            "grid.linestyle":     "--",
            "font.size":          11,
        })
    
        LEGEND_FONT = 11
        LABEL_FONT  = 14
        LW          = 2.5
    
        # (full-course color, partial-course color) per compartment
        COLORS = {
            "gut":    ("#9B2AC4", "#D4A0E8"),
            "blood":  ("#C0392B", "#E89A91"),
            "kidney": ("#7B5230", "#C4A882"),
        }
        GREY = "#888888"
    
        # ── Precompute trajectories ────────────────────────────────────────────────────
    
        ts = np.linspace(-T, 0, 1000)
        tt = ts + T
    
        x_full = np.array([clRA.x(t)      for t in ts])
        x_cons = np.array([clRA_cons.x(t) for t in ts])
        u_full = np.array([clRA.u(t)      for t in ts])
        u_cons = np.array([clRA_cons.u(t) for t in ts])
        d_full = np.array([clRA.d(t)      for t in ts])
        d_cons = np.array([clRA_cons.d(t) for t in ts])
        om     = np.array([omega(t)        for t in ts])
    
        # ── Figure ─────────────────────────────────────────────────────────────────────
    
        fig, axs = plt.subplots(3, 2, figsize=(10, 9), constrained_layout=True)
    
        # Left column – compartment concentrations
        compartments = [
            (0, r"$\mathbf{x}_1(\tau)$", "Drug Concentration (Gut)",     "gut"),
            (1, r"$\mathbf{x}_2(\tau)$", "Drug Concentration (Blood)",   "blood"),
            (2, r"$\mathbf{x}_3(\tau)$", "Drug Concentration (Kidneys)", "kidney"),
        ]
        for row, (i, ylabel, title, ckey) in enumerate(compartments):
            ax = axs[row, 0]
            cf, cp = COLORS[ckey]
            ax.plot(tt, x_full[:, i], color=cf, linewidth=LW, label="Full course")
            ax.plot(tt, x_cons[:, i], color=cp, linewidth=LW, linestyle=":", label="Partial course")
            style_ax(ax, ylabel, title, ylim=(-0.1, 4.1))
    
        axs[0, 0].legend(ncol=1, fontsize=LEGEND_FONT, loc="upper left")
        axs[1, 0].axhline(y=l_x, linestyle="--", color="#27AE60", linewidth=LW, label="Therap. Thresh.")
        axs[1, 0].legend(fontsize=LEGEND_FONT)
        axs[2, 0].axhline(y=g_z, linestyle="--", color="#2C3E50", linewidth=LW, label="Toxic Thresh.")
        axs[2, 0].legend(fontsize=LEGEND_FONT)
    
        # Right column (rows 0–1) – dosing rates
        dosing = [
            (0, 5, r"$\omega_1(\tau)\,\mathbf{u}_1^*(\tau)$", "Dosing Rate (Oral)", "gut"),
            (1, 1, r"$\omega_2(\tau)\,\mathbf{u}_2^*(\tau)$", "Dosing Rate (IV)",   "blood"),
        ]
        for row, (i, scale, ylabel, title, ckey) in enumerate(dosing):
            ax = axs[row, 1]
            cf, cp = COLORS[ckey]
            ax.plot(tt, scale * om * u_full[:, i], color=cf, linewidth=LW)
            ax.plot(tt, scale * om * u_cons[:, i], color=cp, linewidth=LW, linestyle=":")
            ax.plot(tt, scale * om, color=GREY, linewidth=LW, linestyle="--", label=rf"$\omega_{i+1}$")
            style_ax(ax, ylabel, title)
            ax.legend(fontsize=LEGEND_FONT + 3)
    
        # Right column (row 2) – worst-case unmodeled dynamics
        ax = axs[2, 1]
        for i, (ckey, lbl) in enumerate(zip(
            ["gut", "blood", "kidney"],
            [r"$\mathbf{d}_1$", r"$\mathbf{d}_2$", r"$\mathbf{d}_3$"],
        )):
            cf, cp = COLORS[ckey]
            ax.plot(tt, d_full[:, i], color=cf, linewidth=LW, label=lbl)
            ax.plot(tt, d_cons[:, i], color=cp, linewidth=LW, linestyle=":")
    
        style_ax(ax, r"$\mathbf{d}^*(\tau)$", "Worst-Case Unmodeled Dynamics", ylim=(-0.1, 2.1))
        ax.legend(fontsize=LEGEND_FONT, loc="upper left", facecolor="white", framealpha=1)
        plt.savefig("ra.pdf", bbox_inches="tight")

    _()
    plt.show()
    return


@app.cell
def _(T, W_end, W_start, clRAA, g_z, l_x, np, plt):
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
    
        plt.rcParams.update({
            "axes.spines.top":    False,
            "axes.spines.right":  False,
            "axes.grid":          True,
            "grid.alpha":         0.25,
            "grid.linestyle":     "--",
            "font.size":          11,
        })
    
        LEGEND_FONT = 11
        LABEL_FONT  = 14
        LW          = 2.5
    
        # (full-course color, partial-course color) per compartment
        COLORS = {
            "gut":    ("#9B2AC4", "#D4A0E8"),
            "blood":  ("#C0392B", "#E89A91"),
            "kidney": ("#7B5230", "#C4A882"),
        }
        GREY = "#888888"
    
        # ── Precompute trajectories ────────────────────────────────────────────────────
    
        ts = np.linspace(-T, 0, 1000)
        tt = ts + T
    
        x_full = np.array([clRAA.x(t)      for t in ts])
        u_full = np.array([clRAA.u(t)      for t in ts])
        d_full = np.array([clRAA.d(t)      for t in ts])
        om     = np.array([omega(t)        for t in ts])
    
        # ── Figure ─────────────────────────────────────────────────────────────────────
    
        fig, axs = plt.subplots(3, 2, figsize=(10, 9), constrained_layout=True)
    
        # Left column – compartment concentrations
        compartments = [
            (0, r"$\mathbf{x}_1(\tau)$", "Drug Concentration (Gut)",     "gut"),
            (1, r"$\mathbf{x}_2(\tau)$", "Drug Concentration (Blood)",   "blood"),
            (2, r"$\mathbf{x}_3(\tau)$", "Drug Concentration (Kidneys)", "kidney"),
        ]
        for row, (i, ylabel, title, ckey) in enumerate(compartments):
            ax = axs[row, 0]
            cf, cp = COLORS[ckey]
            ax.plot(tt, x_full[:, i], color=cf, linewidth=LW)
            style_ax(ax, ylabel, title, ylim=(-0.1, 1.1))
    
        axs[1, 0].axhline(y=l_x, linestyle="--", color="#27AE60", linewidth=LW, label="Therap. Thresh.")
        axs[1, 0].legend(fontsize=LEGEND_FONT)
        axs[2, 0].axhline(y=g_z, linestyle="--", color="#2C3E50", linewidth=LW, label="Toxic Thresh.")
        axs[2, 0].legend(fontsize=LEGEND_FONT)
    
        # Right column (rows 0–1) – dosing rates
        dosing = [
            (0, 5, r"$\omega_1(\tau)\,\mathbf{u}_1^*(\tau)$", "Dosing Rate (Oral)", "gut"),
            (1, 1, r"$\omega_2(\tau)\,\mathbf{u}_2^*(\tau)$", "Dosing Rate (IV)",   "blood"),
        ]
        for row, (i, scale, ylabel, title, ckey) in enumerate(dosing):
            ax = axs[row, 1]
            cf, cp = COLORS[ckey]
            ax.plot(tt, scale * om * u_full[:, i], color=cf, linewidth=LW)
            ax.plot(tt, scale * om, color=GREY, linewidth=LW, linestyle="--", label=rf"$\omega_{i+1}$")
            style_ax(ax, ylabel, title)
            ax.legend(fontsize=LEGEND_FONT + 3)
    
        # Right column (row 2) – worst-case unmodeled dynamics
        ax = axs[2, 1]
        for i, (ckey, lbl) in enumerate(zip(
            ["gut", "blood", "kidney"],
            [r"$\mathbf{d}_1$", r"$\mathbf{d}_2$", r"$\mathbf{d}_3$"],
        )):
            cf, cp = COLORS[ckey]
            ax.plot(tt, d_full[:, i], color=cf, linewidth=LW, label=lbl)
    
        style_ax(ax, r"$\mathbf{d}^*(\tau)$", "Worst-Case Unmodeled Dynamics", ylim=(-0.1, 2.1))
        ax.legend(fontsize=LEGEND_FONT, loc="upper left", facecolor="white", framealpha=1)
        plt.savefig("raa.pdf", bbox_inches="tight")

    _()
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
