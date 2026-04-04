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
    from dynamics import binding
    from util import closed_loop

    # plotting
    import matplotlib.pyplot as plt

    plt.rcParams["text.usetex"] = False
    plt.rcParams["pdf.fonttype"] = 42  # TrueType
    plt.rcParams["ps.fonttype"] = 42
    font = {"size": 12}
    plt.rc("font", **font)
    return binding, closed_loop, hj, jnp, mo, np, plt


@app.cell
def _(binding, hj, np):
    # This is where all of the work of HJR happens, namely computing the value function V

    # specify the time horizon of the problem
    T = 3

    # specify the weening period
    # W_start = 6  # start weening
    # W_end = 8  # stop dosing

    # specify the dynamics we are considering

    model = binding.binding_model(
        uMax=3.0,
        uMin=0.0,
        dR=0.5,
        dC=0.0,
        kf=2.5,
        kr=0.5,
        gammaX=2.0,
        gammaY=2.1,
        gammaXY=1.5,
    )

    # specify the number of voxels to divide the spatial and temporal axes
    x_voxels = 50
    y_voxels = 50
    z_voxels = 50
    t_voxels = 100

    # Specify bounds
    x_min = -0.1
    y_min = -0.1
    z_min = -0.1

    x_max = +1.5
    y_max = +1.5
    z_max = +3.0

    # Specify therapeutic and toxic thresholds
    l1_x = 1.0
    l2_x = 1.0

    # discretize state-space and the time to solve the HJ Partial Differential Equation
    # don't change
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box([x_min, y_min, z_min], [x_max, y_max, z_max]),
        [x_voxels + 1, y_voxels + 1, z_voxels + 1],
    )

    times = np.linspace(0.0, -T, t_voxels + 1)

    # specify the goal
    l1 = l1_x - grid.states[..., 0]
    l2 = l2_x - grid.states[..., 1]
    return T, grid, l1, l1_x, l2, l2_x, model, times


@app.cell
def _(l1, l2, mo):
    get_plot_fields, set_plot_fields = mo.state({"l1": l1, "l2": l2})
    return get_plot_fields, set_plot_fields


@app.cell(hide_code=True)
def _(get_plot_fields, mo, times):
    plot_fields = get_plot_fields()
    options = list(plot_fields.keys())

    field_name = mo.ui.dropdown(
        options=options,
        value=options[0],
        label="function to plot",
    )
    time_index = mo.ui.slider(
        start=0,
        stop=len(times) - 1,
        step=1,
        value=0,
        label="Time index (for V only)",
    )
    controls = mo.vstack([mo.hstack([field_name, time_index])])
    controls
    return field_name, time_index


@app.cell(hide_code=True)
def _(field_name, get_plot_fields, grid, np, plt, time_index, times):
    import matplotlib.colors as mcolors

    field_map = get_plot_fields()

    field = np.asarray(field_map[field_name.value])
    x_coords = np.asarray(grid.coordinate_vectors[0])
    y_coords = np.asarray(grid.coordinate_vectors[1])
    z_coords = np.asarray(grid.coordinate_vectors[2])
    z_index = int(np.argmin(np.abs(z_coords)))
    z_value = float(z_coords[z_index])

    if field.ndim == 4:
        t_index = int(time_index.value)
        field_2d = field[t_index, :, :, z_index]
        title = (
            f"{field_name.value} at t={float(times[t_index]):.2f}, z={z_value:.2f}"
        )
    else:
        field_2d = field[:, :, z_index]
        title = f"{field_name.value} at z={z_value:.2f}"

    X, Y = np.meshgrid(x_coords, y_coords, indexing="ij")
    vmax = float(np.max(np.abs(field_2d)))

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    if vmax > 0.0:
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        contour = ax.contourf(X, Y, field_2d, levels=31, cmap="RdBu_r", norm=norm)
    else:
        contour = ax.contourf(X, Y, field_2d, levels=31, cmap="RdBu_r")

    ax.contour(X, Y, field_2d, levels=[0.0], colors="black", linewidths=1.0)
    ax.set_xlabel(r"$$[\mathbf{X}]$")
    ax.set_ylabel(r"$$[\mathbf{Y}]$")
    ax.set_title(title)
    fig.colorbar(contour, ax=ax, label=field_name.value)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## First, solve two std R problems, and then two fixed-order RR problems
    """)
    return


@app.cell
def _(grid, hj, jnp, l1, l2, model, set_plot_fields, times):
    def _():

        # specify the reach problem
        def vppR(t, v, l):
            return jnp.minimum(v, l)

        # specify the accuracy with which to solve the HJ Partial Differential Equation
        solver_settings = hj.SolverSettings.with_accuracy(
            "very_high", value_postprocessor=lambda t, v: vppR(t, v, l1)
        )

        # solve for the value function
        V1 = hj.solve(
            hj.SolverSettings.with_accuracy(
                "very_high", value_postprocessor=lambda t, v: vppR(t, v, l1)
            ),
            model,
            grid,
            times,
            l1,
        )
        V2 = hj.solve(
            hj.SolverSettings.with_accuracy(
                "very_high", value_postprocessor=lambda t, v: vppR(t, v, l2)
            ),
            model,
            grid,
            times,
            l2,
        )

        Vboth = hj.solve(
            hj.SolverSettings.with_accuracy(
                "very_high",
                value_postprocessor=lambda t, v: vppR(t, v, jnp.maximum(l1, l2)),
            ),
            model,
            grid,
            times,
            jnp.maximum(l1, l2),
        )

        return V1, V2, Vboth


    VR1, VR2, Vboth = _()
    set_plot_fields(lambda fields: {**fields, "VR1": VR1, "VR2": VR2})
    return VR1, VR2


@app.cell
def _(grid, hj, jnp, l1, l2, model, set_plot_fields, times):
    def _():

        # specify the reach problem
        def vppR(t, v, l):
            return jnp.minimum(v, l)

        # specify the accuracy with which to solve the HJ Partial Differential Equation
        solver_settings = hj.SolverSettings.with_accuracy(
            "very_high", value_postprocessor=lambda t, v: vppR(t, v, l1)
        )

        Vboth = hj.solve(
            hj.SolverSettings.with_accuracy(
                "very_high",
                value_postprocessor=lambda t, v: vppR(t, v, jnp.maximum(l1, l2)),
            ),
            model,
            grid,
            times,
            jnp.maximum(l1, l2),
        )

        return Vboth


    VRboth = _()
    set_plot_fields(lambda fields: {**fields, "VRboth": VRboth})
    return (VRboth,)


@app.cell
def _(VR1, VR2, grid, hj, jnp, l1, l2, model, set_plot_fields, times):
    def _(VR1, VR2):
        # specify the reach i -> reach j problem
        def vppRiRj(t, v, l, times, VR):
            i = jnp.argmin(jnp.abs(t - times))
            return jnp.minimum(v, jnp.maximum(l, VR[i, ...]))

        V12 = hj.solve(
            hj.SolverSettings.with_accuracy(
                "very_high",
                value_postprocessor=lambda t, v: vppRiRj(t, v, l1, times, VR2),
            ),
            model,
            grid,
            times,
            jnp.maximum(l1, VR2[0, ...]),
        )
        V21 = hj.solve(
            hj.SolverSettings.with_accuracy(
                "very_high",
                value_postprocessor=lambda t, v: vppRiRj(t, v, l2, times, VR1),
            ),
            model,
            grid,
            times,
            jnp.maximum(l2, VR1[0, ...]),
        )

        return V12, V21


    V12, V21 = _(VR1, VR2)
    set_plot_fields(lambda fields: {**fields, "V12": V12, "V21": V21})
    return V12, V21


@app.cell
def _(V12, V21, VR1, VR2, VRboth, closed_loop, grid, model, times):
    # Form the closed-loop trajectory

    dummyV = 0 * VR1

    clR1 = closed_loop.ClosedLoopTrajectory(
        model, grid, times, VR1, dummyV, initial_state=[0.0, 0.0, 0.0], steps=100
    )
    clR2 = closed_loop.ClosedLoopTrajectory(
        model, grid, times, VR2, dummyV, initial_state=[0.0, 0.0, 0.0], steps=100
    )
    clRboth = closed_loop.ClosedLoopTrajectory(
        model,
        grid,
        times,
        VRboth,
        dummyV,
        initial_state=[0.0, 0.0, 0.0],
        steps=100,
    )
    clRboth = closed_loop.ClosedLoopTrajectory(
        model,
        grid,
        times,
        VRboth,
        dummyV,
        initial_state=[0.0, 0.0, 0.0],
        steps=100,
    )

    clR12 = closed_loop.ClosedLoopTrajectory(
        model, grid, times, V12, VR2, initial_state=[0.0, 0.0, 0.0], steps=100
    )
    clR21 = closed_loop.ClosedLoopTrajectory(
        model, grid, times, V21, VR1, initial_state=[0.0, 0.0, 0.0], steps=100
    )
    return clR12, clR21, clRboth


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next, solve the RR (with VR1 & VR2)
    """)
    return


@app.cell
def _(VR1, VR2, grid, hj, jnp, l1, l2, model, np, set_plot_fields, times):
    def _(VR1, VR2):

        # specify the reach-reach problem
        def vppRR(t, v, l1, l2, times, VR1, VR2):
            i = jnp.argmin(jnp.abs(t - times))
            return jnp.minimum(
                v,
                jnp.minimum(
                    jnp.maximum(l1, VR2[i, ...]), jnp.maximum(l2, VR1[i, ...])
                ),
            )

        # specify the accuracy with which to solve the HJ Partial Differential Equation
        solver_settings = hj.SolverSettings.with_accuracy(
            "very_high",
            value_postprocessor=lambda t, v: vppRR(t, v, l1, l2, times, VR1, VR2),
        )

        # solve for the value function
        V = hj.solve(
            solver_settings,
            model,
            grid,
            times,
            np.maximum(l1, l2),
        )

        return V


    VRR = _(VR1, VR2)
    set_plot_fields(lambda fields: {**fields, "VRR": VRR})
    return (VRR,)


@app.cell
def _(VR1, VR2, VRR, closed_loop, grid, l1, l2, model, times):
    # Form the closed-loop trajectory

    clRR = closed_loop.ClosedLoopTrajectoryRR(
        model,
        grid,
        times,
        VRR,
        VR1,
        VR2,
        l1,
        l2,
        initial_state=[0.0, 0.0, 0.0],
        steps=100,
    )
    return (clRR,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## We now build the main figures for the paper
    """)
    return


@app.cell
def _(T, clR12, clR21, clRR, l1_x, l2_x, np, plt):
    def _():

        time_var = r"$\tau$"

        # ── Helpers ────────────────────────────────────────────────────────────────────

        # def omega(t):
        #     S1, S2 = W_start - T, W_end - T
        #     return np.clip(1.0 - (t - S1) / (S2 - S1), 0.0, 1.0)

        def style_ax(ax, ylabel, title, xlim=(0, T + 0.1), ylim=None):
            ax.set_xlabel(time_var, fontsize=LABEL_FONT)
            ax.set_ylabel(ylabel, fontsize=LABEL_FONT)
            ax.set_title(title, fontsize=LABEL_FONT)
            # ax.set_xlim(xlim)
            # if ylim is not None:
            #     ax.set_ylim(ylim)

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
            "X": ("blue", "green", "lightblue"),
            "Y": ("blue", "green", "lightblue"),
            "X-Y": ("blue", "green", "lightblue"),
        }
        GREY = "#888888"

        # ── Precompute trajectories ────────────────────────────────────────────────────

        ts = np.linspace(-T, 0, 1000)
        tt = ts + T

        x_r1 = np.array([clR12.x(t) for t in ts])
        x_r2 = np.array([clR21.x(t) for t in ts])
        x_rr = np.array([clRR.x(t) for t in ts])
        u_r1 = np.array([clR12.u(t) for t in ts])
        u_r2 = np.array([clR21.u(t) for t in ts])
        u_rr = np.array([clRR.u(t) for t in ts])
        d_r1 = np.array([clR12.d(t) for t in ts])
        d_r2 = np.array([clR21.d(t) for t in ts])
        d_rr = np.array([clRR.d(t) for t in ts])
        # om = np.array([omega(t) for t in ts])

        # ── Figure ─────────────────────────────────────────────────────────────────────

        fig, axs = plt.subplots(3, 2, figsize=(10, 9), constrained_layout=True)

        # Left column – compartment concentrations
        compartments = [
            (0, r"$[\mathbf{X}](\tau)$", "[X]", "X", 2),
            (1, r"$[\mathbf{Y}](\tau)$", "[Y]", "Y", 2),
            (2, r"$[\mathbf{X-Y}](\tau)$", "[X-Y]", "X-Y", 2),
        ]
        for row, (i, ylabel, title, ckey, ylim) in enumerate(compartments):
            ax = axs[row, 0]
            cf, cp, cr = COLORS[ckey]
            ax.plot(
                tt,
                x_r1[:, i],
                color=cf,
                linewidth=LW,
                label="R1->R2",
                linestyle=":",
            )
            ax.plot(
                tt,
                x_r2[:, i],
                color=cp,
                linewidth=LW,
                label="R2->R1",
                linestyle="-.",
            )
            ax.plot(
                tt, x_rr[:, i], color=cr, linewidth=LW, linestyle="-", label="RR"
            )
            style_ax(ax, ylabel, title, ylim=(-0.1, ylim + 0.1))

        axs[0, 0].legend(ncol=1, fontsize=LEGEND_FONT, loc="upper left")
        axs[0, 0].axhline(
            y=l1_x,
            linestyle="--",
            color="#1F6FCB",
            linewidth=LW,
            label="Thera. Thresh 1",
        )
        axs[1, 0].legend(fontsize=LEGEND_FONT)
        axs[1, 0].axhline(
            y=l2_x,
            linestyle="--",
            color="#27AE60",
            linewidth=LW,
            label="Thera. Thresh 2",
        )
        axs[2, 0].legend(fontsize=LEGEND_FONT)

        # Right column (rows 0–1) – dosing rates
        dosing = [
            (
                0,
                1,
                r"$\mathbf{u}^*_1(\tau)$",
                "[X]",
                "X",
            ),
            (
                1,
                1,
                r"$\mathbf{u}^*_2(\tau)$",
                "[Y]",
                "Y",
            ),
        ]
        for row, (i, scale, ylabel, title, ckey) in enumerate(dosing):
            ax = axs[row, 1]
            cf, cp, cr = COLORS[ckey]
            ax.plot(
                tt,
                scale * u_r1[:, i],
                color=cf,
                linewidth=LW,
                linestyle=":",
            )
            ax.plot(
                tt,
                scale * u_r2[:, i],
                color=cp,
                linewidth=LW,
                linestyle="-.",
            )
            ax.plot(
                tt,
                scale * u_rr[:, i],
                color=cr,
                linewidth=LW,
                linestyle="-",
            )
            style_ax(ax, ylabel, title)
            ax.legend(fontsize=LEGEND_FONT + 3)

        # # Right column (row 2) – worst-case unmodeled dynamics
        # for i, (ckey, ylabel, title) in enumerate(
        #     zip(
        #         ["blood", "kidneys"],
        #         [r"$\mathbf{d}_1(\tau)$", r"$\mathbf{d}_2(\tau)$"],
        #         [
        #             "Worst-Case Transport Disturbance",
        #             "Worst-Case Elimination Disturbance",
        #         ],
        #     )
        # ):
        #     ax = axs[i + 1, 1]
        #     cf, cp, cr = COLORS[ckey]
        #     ax.plot(tt, d_r1[:, i], color=cf, linewidth=LW, linestyle=":")
        #     ax.plot(tt, d_r2[:, i], color=cp, linewidth=LW, linestyle="-.")
        #     ax.plot(tt, d_r2[:, i], color=cr, linewidth=LW, linestyle="-")

        #     style_ax(ax, ylabel, title, ylim=(-0.1, 2.1))


    _()
    plt.savefig("rr.pdf")
    plt.show()
    return


@app.cell
def _(T, clRR, clRboth, l1_x, l2_x, np, plt):
    # Two version
    def _():

        time_var = r"$\tau$"

        # ── Helpers ────────────────────────────────────────────────────────────────────

        # def omega(t):
        #     S1, S2 = W_start - T, W_end - T
        #     return np.clip(1.0 - (t - S1) / (S2 - S1), 0.0, 1.0)

        def style_ax(ax, ylabel, title, xlim=(0, T + 0.5), ylim=None):
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
            "X": ("black", "black", "lightblue"),
            "Y": ("black", "black", "lightblue"),
            "X-Y": ("black", "black", "lightblue"),
        }
        GREY = "#888888"

        # ── Precompute trajectories ────────────────────────────────────────────────────

        ts = np.linspace(-T, 0.5, 1000)
        tt = ts + T

        # x_r1 = np.array([clR12.x(t) for t in ts])
        # x_r2 = np.array([clR21.x(t) for t in ts])
        # x_rr = np.array([clRR.x(t) for t in ts])
        # u_r1 = np.array([clR12.u(t) for t in ts])
        # u_r2 = np.array([clR21.u(t) for t in ts])
        # u_rr = np.array([clRR.u(t) for t in ts])
        # d_r1 = np.array([clR12.d(t) for t in ts])
        # d_r2 = np.array([clR21.d(t) for t in ts])
        # d_rr = np.array([clRR.d(t) for t in ts])
        # # om = np.array([omega(t) for t in ts])

        x_r1 = np.array([clRboth.x(t) for t in ts])
        # x_r1 = np.array([clR1.x(t) for t in ts])
        # x_r2 = np.array([clR2.x(t) for t in ts])
        x_rr = np.array([clRR.x(t) for t in ts])
        u_r1 = np.array([clRboth.u(t) for t in ts])
        # u_r2 = np.array([clR2.u(t) for t in ts])
        u_rr = np.array([clRR.u(t) for t in ts])
        d_r1 = np.array([clRboth.d(t) for t in ts])
        # d_r2 = np.array([clR2.d(t) for t in ts])
        d_rr = np.array([clRR.d(t) for t in ts])
        # om = np.array([omega(t) for t in ts])

        # ── Figure ─────────────────────────────────────────────────────────────────────

        fig, axs = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)

        # Left column – compartment concentrations
        compartments = [
            (0, r"$\mathbf{x}_1(\tau)$", r"Concentration of $\mathrm{X}_1$", "X", 1.7),
            (1, r"$\mathbf{x}_2(\tau)$", r"Concentration of $\mathrm{X}_2$", "Y", 1.7),
            # (2, r"$[\mathbf{X-Y}](\tau)$", r"$[\mathbf{X-Y}](\tau)$", "X-Y", 0.5),
        ]
        for plti, (i, ylabel, title, ckey, ylim) in enumerate(compartments):
            if plti < 2:
                # ax = axs[0, plti]
                ax = axs[plti]
            else:
                ax = axs[1, 0]
            cf, cp, cr = COLORS[ckey]
            ax.plot(
                tt, x_rr[:, i], color=cr, linewidth=LW, linestyle="-", label="RR"
            )
            ax.plot(
                tt,
                x_r1[:, i],
                color=cf,
                linewidth=LW,
                label="R simultaneous",
                linestyle="-.",
            )
            # ax.plot(
            #     tt, x_r2[:, i], color=cp, linewidth=LW, label="R2", linestyle="-.",
            # )
            style_ax(ax, ylabel, title, ylim=(-0.1, ylim + 0.1))

        # axs[0].axhline(
        #     y=l1_x,
        #     linestyle="--",
        #     color="#1F6FCB",
        #     linewidth=LW,
        #     label="Therap. Thresh. 1",
        # )
        # axs[0].legend(ncol=1, fontsize=LEGEND_FONT, loc="upper left")
        # axs[1].axhline(
        #     y=l2_x,
        #     linestyle="--",
        #     color="#27AE60",
        #     linewidth=LW,
        #     label="Therap. Thresh. 2",
        # )
        # axs[1].legend(fontsize=LEGEND_FONT)

        axs[0].axhline(
            y=l1_x,
            linestyle="--",
            color="#1F6FCB",
            linewidth=LW,
            label=r"Therap. Thresh. $\theta_\mathrm{ther,1}$",
        )
        axs[0].legend(ncol=1, fontsize=LEGEND_FONT, loc="upper left")
        axs[1].axhline(
            y=l2_x,
            linestyle="--",
            color="#27AE60",
            linewidth=LW,
            label=r"Therap. Thresh. $\theta_\mathrm{ther,2}$",
        )
        axs[1].legend(fontsize=LEGEND_FONT, loc="upper left")
        # axs[1, 0].legend(fontsize=LEGEND_FONT)

        # field = np.asarray(field_map[field_name.value])
        # x_coords = np.asarray(grid.coordinate_vectors[0])
        # y_coords = np.asarray(grid.coordinate_vectors[1])
        # z_coords = np.asarray(grid.coordinate_vectors[2])
        # z_index = int(np.argmin(np.abs(z_coords)))
        # z_value = float(z_coords[z_index])

        # t_index = 100
        # value_2d = VRR[t_index, :, :, z_index]
        # title = (
        #     r"$V_{RR}\left([\mathbf{X}], [\mathbf{Y}], [\mathbf{X-Y}]=0, t=-3\right)$"
        # )
        # print("t=",times[t_index])

        # X, Y = np.meshgrid(x_coords, y_coords, indexing="ij")
        # vmax = float(np.max(np.abs(value_2d)))

        # ax = axs[1, 1]
        # if vmax > 0.0:
        #     norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        #     contour = ax.contourf(X, Y, -value_2d, levels=31, cmap="RdBu", norm=norm)
        #     cbar_ticks = [-vmax, vmax]
        # else:
        #     contour = ax.contourf(X, Y, -value_2d, levels=31, cmap="RdBu")
        #     cbar_ticks = [value_2d.min(), value_2d.max()]

        # print(cbar_ticks)

        # # ax.contour(X, Y, -value_2d, levels=[0], colors="black", linewidths=2.0)
        # ax.set_xlabel(r"$\mathbf{x}_1$")
        # ax.set_ylabel(r"$\mathbf{x}_2$")
        # ax.set_title(title)

        # cbar = fig.colorbar(contour, ax=ax, label=r"$V_{RR}$", ticks=cbar_ticks)
        # cbar = fig.colorbar(contour, ax=ax, ticks=[0])
        # cbar = fig.colorbar(contour, ax=ax, label=r"$V_{RR}$", ticks=[0.15, 0.5])
        # from matplotlib.ticker import FormatStrFormatter
        # cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        # cbar.set_label(r"$V_{RR}$", rotation=0, labelpad=-25)
        # cbar.ax.yaxis.set_label_coords(3.0, 0.5)  # x, y in axes coords


    _()
    plt.savefig("/Users/dylanhirsch/Desktop/rr2.pdf")
    plt.show()
    return


@app.cell
def _(T, clRR, clRboth, l1_x, l2_x, np, plt):
    # Three version
    def _():

        time_var = r"$\tau$"

        # ── Helpers ────────────────────────────────────────────────────────────────────

        # def omega(t):
        #     S1, S2 = W_start - T, W_end - T
        #     return np.clip(1.0 - (t - S1) / (S2 - S1), 0.0, 1.0)

        def style_ax(ax, ylabel, title, xlim=(0, T + 0.5), ylim=None):
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
            "X": ("black", "black", "lightblue"),
            "Y": ("black", "black", "lightblue"),
            "X-Y": ("black", "black", "lightblue"),
        }
        GREY = "#888888"

        # ── Precompute trajectories ────────────────────────────────────────────────────

        ts = np.linspace(-T, 0.5, 1000)
        tt = ts + T

        # x_r1 = np.array([clR12.x(t) for t in ts])
        # x_r2 = np.array([clR21.x(t) for t in ts])
        # x_rr = np.array([clRR.x(t) for t in ts])
        # u_r1 = np.array([clR12.u(t) for t in ts])
        # u_r2 = np.array([clR21.u(t) for t in ts])
        # u_rr = np.array([clRR.u(t) for t in ts])
        # d_r1 = np.array([clR12.d(t) for t in ts])
        # d_r2 = np.array([clR21.d(t) for t in ts])
        # d_rr = np.array([clRR.d(t) for t in ts])
        # # om = np.array([omega(t) for t in ts])

        x_r1 = np.array([clRboth.x(t) for t in ts])
        # x_r1 = np.array([clR1.x(t) for t in ts])
        # x_r2 = np.array([clR2.x(t) for t in ts])
        x_rr = np.array([clRR.x(t) for t in ts])
        u_r1 = np.array([clRboth.u(t) for t in ts])
        # u_r2 = np.array([clR2.u(t) for t in ts])
        u_rr = np.array([clRR.u(t) for t in ts])
        d_r1 = np.array([clRboth.d(t) for t in ts])
        # d_r2 = np.array([clR2.d(t) for t in ts])
        d_rr = np.array([clRR.d(t) for t in ts])
        # om = np.array([omega(t) for t in ts])

        # ── Figure ─────────────────────────────────────────────────────────────────────

        fig, axs = plt.subplots(1, 3, figsize=(10, 4), constrained_layout=True)

        # Left column – compartment concentrations
        compartments = [
            (0, r"$\mathbf{x}_1(\tau)$", r"$\mathbf{x}_1(\tau)$", "X", 1.5),
            (1, r"$\mathbf{x}_2(\tau)$", r"$\mathbf{x}_2(\tau)$", "Y", 1.5),
            (2, r"$\mathbf{x}_3(\tau)$", r"$\mathbf{x}_3(\tau)$", "X-Y", 1.0),
        ]
        for plti, (i, ylabel, title, ckey, ylim) in enumerate(compartments):
            ax = axs[plti]
            cf, cp, cr = COLORS[ckey]
            ax.plot(
                tt, x_rr[:, i], color=cr, linewidth=LW, linestyle="-", label="RR"
            )
            ax.plot(
                tt,
                x_r1[:, i],
                color=cf,
                linewidth=LW,
                label="R simultaneous",
                linestyle="-.",
            )
            # ax.plot(
            #     tt, x_r2[:, i], color=cp, linewidth=LW, label="R2", linestyle="-.",
            # )
            style_ax(ax, ylabel, title, ylim=(-0.1, ylim + 0.1))

        # axs[0].axhline(
        #     y=l1_x,
        #     linestyle="--",
        #     color="#1F6FCB",
        #     linewidth=LW,
        #     label="Therap. Thresh. 1",
        # )
        # axs[0].legend(ncol=1, fontsize=LEGEND_FONT, loc="upper left")
        # axs[1].axhline(
        #     y=l2_x,
        #     linestyle="--",
        #     color="#27AE60",
        #     linewidth=LW,
        #     label="Therap. Thresh. 2",
        # )
        # axs[1].legend(fontsize=LEGEND_FONT)

        axs[0].axhline(
            y=l1_x,
            linestyle="--",
            color="#1F6FCB",
            linewidth=LW,
            label="Therap. Thresh. 1",
        )
        axs[0].legend(ncol=1, fontsize=LEGEND_FONT, loc="upper left")
        axs[1].axhline(
            y=l2_x,
            linestyle="--",
            color="#27AE60",
            linewidth=LW,
            label="Therap. Thresh. 2",
        )
        axs[1].legend(fontsize=LEGEND_FONT, loc="upper left")
        axs[2].legend(fontsize=LEGEND_FONT, loc="upper left")

        # field = np.asarray(field_map[field_name.value])
        # x_coords = np.asarray(grid.coordinate_vectors[0])
        # y_coords = np.asarray(grid.coordinate_vectors[1])
        # z_coords = np.asarray(grid.coordinate_vectors[2])
        # z_index = int(np.argmin(np.abs(z_coords)))
        # z_value = float(z_coords[z_index])

        # t_index = 100
        # value_2d = VRR[t_index, :, :, z_index]
        # title = (
        #     r"$V_{RR}\left([\mathbf{X}], [\mathbf{Y}], [\mathbf{X-Y}]=0, t=-3\right)$"
        # )
        # print("t=",times[t_index])

        # X, Y = np.meshgrid(x_coords, y_coords, indexing="ij")
        # vmax = float(np.max(np.abs(value_2d)))

        # ax = axs[1, 1]
        # if vmax > 0.0:
        #     norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        #     contour = ax.contourf(X, Y, -value_2d, levels=31, cmap="RdBu", norm=norm)
        #     cbar_ticks = [-vmax, vmax]
        # else:
        #     contour = ax.contourf(X, Y, -value_2d, levels=31, cmap="RdBu")
        #     cbar_ticks = [value_2d.min(), value_2d.max()]

        # print(cbar_ticks)

        # # ax.contour(X, Y, -value_2d, levels=[0], colors="black", linewidths=2.0)
        # ax.set_xlabel(r"$\mathbf{x}_1$")
        # ax.set_ylabel(r"$\mathbf{x}_2$")
        # ax.set_title(title)

        # cbar = fig.colorbar(contour, ax=ax, label=r"$V_{RR}$", ticks=cbar_ticks)
        # cbar = fig.colorbar(contour, ax=ax, ticks=[0])
        # cbar = fig.colorbar(contour, ax=ax, label=r"$V_{RR}$", ticks=[0.15, 0.5])
        # from matplotlib.ticker import FormatStrFormatter
        # cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        # cbar.set_label(r"$V_{RR}$", rotation=0, labelpad=-25)
        # cbar.ax.yaxis.set_label_coords(3.0, 0.5)  # x, y in axes coords


    _()
    plt.savefig("rr3.pdf")
    plt.show()
    return


@app.cell
def _(T, clRR, clRboth, l1_x, l2_x, np, plt):
    # Square version
    def _():

        time_var = r"$\tau$"

        # ── Helpers ────────────────────────────────────────────────────────────────────

        # def omega(t):
        #     S1, S2 = W_start - T, W_end - T
        #     return np.clip(1.0 - (t - S1) / (S2 - S1), 0.0, 1.0)

        def style_ax(ax, ylabel, title, xlim=(0, T + 0.5), ylim=None):
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
            "X": ("black", "black", "lightblue"),
            "Y": ("black", "black", "lightblue"),
            "X-Y": ("black", "black", "lightblue"),
        }
        GREY = "#888888"

        # ── Precompute trajectories ────────────────────────────────────────────────────

        ts = np.linspace(-T, 0.5, 1000)
        tt = ts + T

        # x_r1 = np.array([clR12.x(t) for t in ts])
        # x_r2 = np.array([clR21.x(t) for t in ts])
        # x_rr = np.array([clRR.x(t) for t in ts])
        # u_r1 = np.array([clR12.u(t) for t in ts])
        # u_r2 = np.array([clR21.u(t) for t in ts])
        # u_rr = np.array([clRR.u(t) for t in ts])
        # d_r1 = np.array([clR12.d(t) for t in ts])
        # d_r2 = np.array([clR21.d(t) for t in ts])
        # d_rr = np.array([clRR.d(t) for t in ts])
        # # om = np.array([omega(t) for t in ts])

        x_r1 = np.array([clRboth.x(t) for t in ts])
        # x_r1 = np.array([clR1.x(t) for t in ts])
        # x_r2 = np.array([clR2.x(t) for t in ts])
        x_rr = np.array([clRR.x(t) for t in ts])
        u_r1 = np.array([clRboth.u(t) for t in ts])
        # u_r2 = np.array([clR2.u(t) for t in ts])
        u_rr = np.array([clRR.u(t) for t in ts])
        d_r1 = np.array([clRboth.d(t) for t in ts])
        # d_r2 = np.array([clR2.d(t) for t in ts])
        d_rr = np.array([clRR.d(t) for t in ts])
        # om = np.array([omega(t) for t in ts])

        # ── Figure ─────────────────────────────────────────────────────────────────────

        fig, axs = plt.subplots(2, 2, figsize=(8, 5), constrained_layout=True)

        # Left column – compartment concentrations
        compartments = [
            (0, r"$\mathbf{x}_1(\tau)$", r"$\mathbf{x}_1(\tau)$", "X", 2.5),
            (1, r"$\mathbf{x}_2(\tau)$", r"$\mathbf{x}_2(\tau)$", "Y", 2.5),
            (2, r"$\mathbf{x}_3(\tau)$", r"$\mathbf{x}_3(\tau)$", "X-Y", 1.5),
        ]
        for plti, (i, ylabel, title, ckey, ylim) in enumerate(compartments):
            if plti < 2:
                ax = axs[0, plti]
            else:
                ax = axs[1, 0]
            cf, cp, cr = COLORS[ckey]
            ax.plot(
                tt, x_rr[:, i], color=cr, linewidth=LW, linestyle="-", label="RR"
            )
            ax.plot(
                tt,
                x_r1[:, i],
                color=cf,
                linewidth=LW,
                label="R simultaneous",
                linestyle="-.",
            )
            # ax.plot(
            #     tt, x_r2[:, i], color=cp, linewidth=LW, label="R2", linestyle="-.",
            # )
            style_ax(ax, ylabel, title, ylim=(-0.1, ylim + 0.1))

        # axs[0, 0].axhline(
        #     y=l1_x,
        #     linestyle="--",
        #     color="#1F6FCB",
        #     linewidth=LW,
        #     label="Therap. Thresh. 1",
        # )
        # axs[0, 0].legend(ncol=1, fontsize=LEGEND_FONT, loc="upper left")
        # axs[0, 1].axhline(
        #     y=l2_x,
        #     linestyle="--",
        #     color="#27AE60",
        #     linewidth=LW,
        #     label="Therap. Thresh. 2",
        # )
        # axs[0, 1].legend(fontsize=LEGEND_FONT)

        axs[0, 0].axhline(
            y=l1_x,
            linestyle="--",
            color="#1F6FCB",
            linewidth=LW,
            label=r"Therap. Thresh. \theta_{\mathrm{ther}, 1}",
        )
        axs[0, 0].legend(ncol=1, fontsize=LEGEND_FONT, loc="upper left")
        axs[0, 1].axhline(
            y=l2_x,
            linestyle="--",
            color="#27AE60",
            linewidth=LW,
            label=r"Therap. Thresh. \theta_{\mathrm{ther}, 2}",
        )
        axs[0, 1].legend(fontsize=LEGEND_FONT, loc="upper left")
        axs[1, 0].legend(fontsize=LEGEND_FONT, loc="upper left")

        # field = np.asarray(field_map[field_name.value])
        # x_coords = np.asarray(grid.coordinate_vectors[0])
        # y_coords = np.asarray(grid.coordinate_vectors[1])
        # z_coords = np.asarray(grid.coordinate_vectors[2])
        # z_index = int(np.argmin(np.abs(z_coords)))
        # z_value = float(z_coords[z_index])

        # t_index = 100
        # value_2d = VRR[t_index, :, :, z_index]
        # title = (
        #     r"$V_{RR}\left([\mathbf{X}], [\mathbf{Y}], [\mathbf{X-Y}]=0, t=-3\right)$"
        # )
        # print("t=",times[t_index])

        # X, Y = np.meshgrid(x_coords, y_coords, indexing="ij")
        # vmax = float(np.max(np.abs(value_2d)))

        # ax = axs[1, 1]
        # if vmax > 0.0:
        #     norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        #     contour = ax.contourf(X, Y, -value_2d, levels=31, cmap="RdBu", norm=norm)
        #     cbar_ticks = [-vmax, vmax]
        # else:
        #     contour = ax.contourf(X, Y, -value_2d, levels=31, cmap="RdBu")
        #     cbar_ticks = [value_2d.min(), value_2d.max()]

        # print(cbar_ticks)

        # # ax.contour(X, Y, -value_2d, levels=[0], colors="black", linewidths=2.0)
        # ax.set_xlabel(r"$\mathbf{x}_1$")
        # ax.set_ylabel(r"$\mathbf{x}_2$")
        # ax.set_title(title)

        # cbar = fig.colorbar(contour, ax=ax, label=r"$V_{RR}$", ticks=cbar_ticks)
        # cbar = fig.colorbar(contour, ax=ax, ticks=[0])
        # cbar = fig.colorbar(contour, ax=ax, label=r"$V_{RR}$", ticks=[0.15, 0.5])
        # from matplotlib.ticker import FormatStrFormatter
        # cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        # cbar.set_label(r"$V_{RR}$", rotation=0, labelpad=-25)
        # cbar.ax.yaxis.set_label_coords(3.0, 0.5)  # x, y in axes coords

        # Right column (row 2) – worst-case unmodeled dynamics
        for i, (ckey, ylabel, title) in enumerate(
            zip(
                ["X"],
                [""],
                [
                    "Optimal Inputs",
                ],
            )
        ):
            ax = axs[i + 1, 1]
            cf, cp, cr = COLORS[ckey]
            ax.plot(
                tt,
                u_rr[:, i],
                color=cr,
                linewidth=LW,
                linestyle=":",
                label=r"$\mathbf{u}_1$" + " RR",
            )
            ax.plot(
                tt,
                u_r1[:, i],
                color=cf,
                linewidth=LW,
                linestyle=":",
                label=r"$\mathbf{u}_1$" + " R simultaneous",
            )
            ax.plot(
                tt,
                d_rr[:, i],
                color=cr,
                linewidth=LW,
                linestyle="-",
                label=r"$\mathbf{d}_1$" + " RR",
            )
            ax.plot(
                tt,
                d_r1[:, i],
                color=cf,
                linewidth=LW,
                linestyle="-",
                label=r"$\mathbf{d}_1$" + " R simultaneous",
            )

            style_ax(ax, ylabel, title, ylim=(-0.1, 10.0))
            axs[1, 1].legend(fontsize=LEGEND_FONT, loc="upper left")


    _()
    plt.savefig("rr4.pdf")
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
