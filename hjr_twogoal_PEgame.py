import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Two-key pursuit--evasion game

    Player 1 (the capturer/evader) must complete
    $$(K_1\rightarrow R_1)\;\lor\;(K_2\rightarrow R_2)$$
    while always avoiding capture by Player 2 (the defender/pursuer).

    The arena is $[-5,5]^2$.  The value convention is **negative = winning for
    P1**, and the state is $(x_1,y_1,x_2,y_2)$.
    """)
    return


@app.cell
def _():
    import emoji
    import hj_reachability as hj
    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    from matplotlib.patches import Circle, Rectangle
    import numpy as np

    from dynamics import singleint_twoagent
    from util import closed_loop

    plt.rcParams["text.usetex"] = False
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.size"] = 12
    return (
        AnnotationBbox,
        Circle,
        OffsetImage,
        Rectangle,
        closed_loop,
        emoji,
        hj,
        jnp,
        mo,
        np,
        plt,
        singleint_twoagent,
    )


@app.cell
def _(mo):
    resolution = mo.ui.dropdown(
        options={
            "Sparse / preview (31 points per dimension)": 31,
            "High resolution (51 points per dimension)": 51,
        },
        value="Sparse / preview (31 points per dimension)",
        label="Spatial resolution",
    )
    resolution
    return (resolution,)


@app.cell
def _(hj, jnp, np, resolution, singleint_twoagent):
    # Geometry and numerical configuration.
    arena_half_width = 5.0
    key_radius = 0.5
    capture_radius = 0.5
    key_centers = ((-2.0, 2.25), (-2.0, -2.25))
    goal_centers = ((4.5, 2.5), (4.5, -2.5))
    goal_width, goal_height = 1.0, 5.0

    initial_state = np.array([-3.0, 0.005, 3.0, 0.0])
    horizon = 17.0
    time_points = 31
    times = np.linspace(0.0, -horizon, time_points)

    grid_points = resolution.value
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(
            jnp.full(4, -arena_half_width),
            jnp.full(4, arena_half_width),
        ),
        (grid_points,) * 4,
    )
    model = singleint_twoagent.singleint_twoagent(vmax_u=1.0, vmax_d=0.8)

    p1x, p1y, p2x, p2y = (grid.states[..., i] for i in range(4))

    def circle_sdf(x, y, center, radius):
        return jnp.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2) - radius

    def box_sdf(x, y, center, width, height):
        # Exact signed distance to an axis-aligned rectangle.
        qx = jnp.abs(x - center[0]) - width / 2
        qy = jnp.abs(y - center[1]) - height / 2
        outside = jnp.sqrt(jnp.maximum(qx, 0) ** 2 + jnp.maximum(qy, 0) ** 2)
        inside = jnp.minimum(jnp.maximum(qx, qy), 0)
        return outside + inside

    k1 = circle_sdf(p1x, p1y, key_centers[0], key_radius)
    k2 = circle_sdf(p1x, p1y, key_centers[1], key_radius)
    r1 = box_sdf(p1x, p1y, goal_centers[0], goal_width, goal_height)
    r2 = box_sdf(p1x, p1y, goal_centers[1], goal_width, goal_height)

    # Positive in the unsafe capture set, as required by the max obstacle.
    capture = capture_radius - jnp.sqrt((p1x - p2x) ** 2 + (p1y - p2y) ** 2)
    return (
        capture,
        goal_centers,
        goal_height,
        goal_width,
        grid,
        horizon,
        initial_state,
        k1,
        k2,
        key_centers,
        key_radius,
        model,
        r1,
        r2,
        times,
    )


@app.cell(hide_code=True)
def _(
    AnnotationBbox,
    Circle,
    OffsetImage,
    Rectangle,
    emoji,
    goal_centers,
    goal_height,
    goal_width,
    initial_state,
    key_centers,
    key_radius,
    np,
    plt,
):
    def render_apple_emoji(shortcode, point_size=32, padding=12):
        """Render any emoji-package shortcode with macOS Apple Color Emoji."""
        import importlib as _importlib
        import platform as _platform
        from io import BytesIO as _BytesIO

        from PIL import Image as _Image

        if _platform.system() != "Darwin":
            raise RuntimeError(
                "Apple Color Emoji rendering requires macOS. "
                "Use image markers such as Twemoji on other platforms."
            )

        # `AppKit` is the import name supplied by pyobjc-framework-Cocoa.
        # Import dynamically so Marimo's package manager does not confuse it
        # with the unrelated PyPI distribution named `AppKit`.
        try:
            _appkit = _importlib.import_module("App" + "Kit")
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Apple emoji rendering needs pyobjc-framework-Cocoa. "
                "Run `uv sync` from this repository, or install "
                "`pyobjc-framework-Cocoa` in the active environment. "
                "Do not install the PyPI package named `AppKit`; it is an "
                "unrelated GTK package that requires Cairo."
            ) from exc

        required_symbols = (
            "NSAttributedString",
            "NSBitmapImageFileTypePNG",
            "NSBitmapImageRep",
            "NSFont",
            "NSFontAttributeName",
            "NSImage",
        )
        missing_symbols = [
            name for name in required_symbols if not hasattr(_appkit, name)
        ]
        if missing_symbols:
            module_path = getattr(_appkit, "__file__", "an unknown location")
            raise ImportError(
                f"Loaded an incompatible AppKit module from {module_path}. "
                "Uninstall the PyPI package `AppKit`, then install "
                "`pyobjc-framework-Cocoa`."
            )

        _NSAttributedString = _appkit.NSAttributedString
        _NSBitmapImageFileTypePNG = _appkit.NSBitmapImageFileTypePNG
        _NSBitmapImageRep = _appkit.NSBitmapImageRep
        _NSFont = _appkit.NSFont
        _NSFontAttributeName = _appkit.NSFontAttributeName
        _NSImage = _appkit.NSImage

        glyph = emoji.emojize(shortcode, language="alias")
        if glyph == shortcode:
            raise ValueError(f"Unknown emoji shortcode: {shortcode}")

        font = _NSFont.fontWithName_size_("Apple Color Emoji", point_size)
        if font is None:
            raise RuntimeError("Apple Color Emoji is unavailable on this system")

        attributed = _NSAttributedString.alloc().initWithString_attributes_(
            glyph, {_NSFontAttributeName: font}
        )
        width, height = attributed.size()
        native_image = _NSImage.alloc().initWithSize_(
            (width + 2 * padding, height + 2 * padding)
        )
        native_image.lockFocus()
        attributed.drawAtPoint_((padding, padding))
        native_image.unlockFocus()

        bitmap = _NSBitmapImageRep.imageRepWithData_(native_image.TIFFRepresentation())
        png_data = bitmap.representationUsingType_properties_(
            _NSBitmapImageFileTypePNG, {}
        )
        image = _Image.open(_BytesIO(bytes(png_data))).convert("RGBA")
        alpha_bounds = image.getbbox()
        if alpha_bounds is None:
            raise RuntimeError(f"Apple Color Emoji could not render {shortcode}")
        return np.asarray(image.crop(alpha_bounds))

    # Change these to any alias supported by emoji.emojize(), for example
    # :woman:, :woman_technologist:, or :woman_technologist_medium_skin_tone:.
    p1_emoji_shortcode = ":woman:"
    p2_emoji_shortcode = ":zombie:"
    p1_marker_image = render_apple_emoji(p1_emoji_shortcode)
    p2_marker_image = render_apple_emoji(p2_emoji_shortcode)

    def add_emoji_marker(ax, xy, image, zoom=0.5):
        marker = AnnotationBbox(
            OffsetImage(image, zoom=zoom),
            xy,
            frameon=False,
            pad=0,
            box_alignment=(0.5, 0.5),
            zorder=10,
        )
        ax.add_artist(marker)
        return marker

    def decorate_arena(ax, show_players=True):
        colors = ("tab:purple", "tab:orange")
        for i, (center, color) in enumerate(zip(key_centers, colors), start=1):
            ax.add_patch(
                Circle(
                    center,
                    key_radius,
                    facecolor=color,
                    alpha=0.25,
                    edgecolor=color,
                    linewidth=2,
                )
            )
            ax.annotate(rf"$K_{i}$", center, ha="center", va="center", fontsize=22)
        for i, (center, color) in enumerate(zip(goal_centers, colors), start=1):
            lower_left = (center[0] - goal_width / 2, center[1] - goal_height / 2)
            ax.add_patch(
                Rectangle(
                    lower_left,
                    goal_width,
                    goal_height,
                    facecolor=color,
                    alpha=0.18,
                    edgecolor=color,
                    linewidth=2,
                )
            )
            ax.annotate(rf"$R_{i}$", center, ha="center", va="center", fontsize=22)
        if show_players:
            add_emoji_marker(ax, initial_state[:2], p1_marker_image)
            add_emoji_marker(ax, initial_state[2:], p2_marker_image)
        ax.set(xlim=(-5, 5), ylim=(-5, 5), xlabel=r"$x$", ylabel=r"$y$")
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)

    fig_configuration, ax_configuration = plt.subplots(figsize=(7, 7))
    decorate_arena(ax_configuration)
    ax_configuration.set_title("Two-key pursuit--evasion configuration")
    fig_configuration.tight_layout()
    fig_configuration
    return add_emoji_marker, decorate_arena, p1_marker_image, p2_marker_image


@app.cell
def _(mo):
    solve_values = mo.ui.run_button(label="Solve V3 → V1/V2 → V0")
    solve_values
    return (solve_values,)


@app.cell
def _(mo, solve_values):
    mo.stop(
        not solve_values.value,
        mo.md(
            "The configuration is ready. Press **Solve** to start the four 4D HJ solves."
        ),
    )
    solver_ready = True
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Backward solution order

    Each solve uses the reach--avoid postprocessor
    $$V\leftarrow\max\{\min(V,r),\ c\},$$
    where $c$ is positive in the capture set.  Dependencies are evaluated at
    the closest solver time, so the decomposition remains time consistent.
    """)
    return


@app.cell
def _(capture, grid, hj, initial_state, jnp, np, plt, times):
    time_array = jnp.asarray(times)

    def solve_reach_avoid(model, target_at_time):
        settings = hj.SolverSettings.with_accuracy(
            "high",
            value_postprocessor=lambda t, v: jnp.maximum(
                jnp.minimum(v, target_at_time(t)), capture
            ),
        )
        terminal_target = target_at_time(time_array[0])
        return hj.solve(
            settings, model, grid, times, jnp.maximum(terminal_target, capture)
        )

    def downstream_at(t, values):
        index = jnp.argmin(jnp.abs(t - time_array))
        return values[index, ...]

    p2_x_index = int(
        np.argmin(np.abs(np.asarray(grid.coordinate_vectors[2]) - initial_state[2]))
    )
    p2_y_index = int(
        np.argmin(np.abs(np.asarray(grid.coordinate_vectors[3]) - initial_state[3]))
    )
    x_coordinates = np.asarray(grid.coordinate_vectors[0])
    y_coordinates = np.asarray(grid.coordinate_vectors[1])
    plot_x, plot_y = np.meshgrid(x_coordinates, y_coordinates, indexing="ij")

    def plot_value_slice(values, title, decorate_arena):
        field = np.asarray(values[-1, :, :, p2_x_index, p2_y_index])
        fig, ax = plt.subplots(figsize=(7, 7))
        magnitude = float(np.nanmax(np.abs(field)))
        levels = np.linspace(-magnitude, magnitude, 31) if magnitude > 0 else 31
        filled = ax.contourf(
            plot_x, plot_y, field, levels=levels, cmap="RdBu_r", extend="both"
        )
        if np.nanmin(field) <= 0 <= np.nanmax(field):
            ax.contour(plot_x, plot_y, field, levels=[0], colors="black", linewidths=2)
        decorate_arena(ax)
        ax.set_title(title + r"; P2 fixed at $(4,0)$")
        fig.colorbar(filled, ax=ax, label="value")
        fig.tight_layout()
        return fig

    return downstream_at, plot_value_slice, solve_reach_avoid


@app.cell
def _(decorate_arena, model, plot_value_slice, r1, solve_reach_avoid):
    # V1 only -- if we could only go one of the ways
    V1_only = solve_reach_avoid(model, lambda _t: r1)
    fig_v1_only = plot_value_slice(V1_only, r"$V_{\text{only} 1}$: $r=R_1$", decorate_arena)
    fig_v1_only
    return


@app.cell
def _(decorate_arena, jnp, model, plot_value_slice, r1, r2, solve_reach_avoid):
    # V3: after both keys have been collected, reach either matching goal.
    r3 = jnp.minimum(r1, r2)
    V3 = solve_reach_avoid(model, lambda _t: r3)
    fig_v3 = plot_value_slice(V3, r"$V_3$: $r_3=\max\{R_1,R_2\}$", decorate_arena)
    fig_v3
    return (V3,)


@app.cell
def _(
    V3,
    decorate_arena,
    downstream_at,
    jnp,
    k2,
    model,
    plot_value_slice,
    r1,
    solve_reach_avoid,
):
    # V1: K1 was collected; finish at G1, or collect K2 and continue with V3.
    def target_v1(t):
        return jnp.minimum(r1, jnp.maximum(k2, downstream_at(t, V3)))

    V1 = solve_reach_avoid(model, target_v1)
    fig_v1 = plot_value_slice(
        V1, r"$V_1$: $r_1=\max\{R_1,\min(K_2,V_3)\}$", decorate_arena
    )
    fig_v1
    return (V1,)


@app.cell
def _(
    V3,
    decorate_arena,
    downstream_at,
    jnp,
    k1,
    model,
    plot_value_slice,
    r2,
    solve_reach_avoid,
):
    # V2: K2 was collected; finish at R2, or collect K1 and continue with V3.
    def target_v2(t):
        return jnp.minimum(r2, jnp.maximum(k1, downstream_at(t, V3)))

    V2 = solve_reach_avoid(model, target_v2)
    fig_v2 = plot_value_slice(
        V2, r"$V_2$: $r_2=\max\{R_2,\min(K_1,V_3)\}$", decorate_arena
    )
    fig_v2
    return (V2,)


@app.cell
def _(
    V1,
    V2,
    decorate_arena,
    downstream_at,
    jnp,
    k1,
    k2,
    model,
    plot_value_slice,
    solve_reach_avoid,
):
    # V0: choose either key/goal branch while maintaining capture avoidance.
    def target_v0(t):
        return jnp.minimum(
            jnp.maximum(k1, downstream_at(t+1, V1)),
            jnp.maximum(k2, downstream_at(t+1, V2)),
        )

    V0 = solve_reach_avoid(model, target_v0)
    fig_v0 = plot_value_slice(
        V0,
        r"$V_0$: $r_0=\max\{\min(K_1,V_1),\min(K_2,V_2)\}$",
        decorate_arena,
    )
    fig_v0
    return (V0,)


@app.cell
def _(
    V0,
    V1,
    V2,
    V3,
    capture,
    closed_loop,
    grid,
    initial_state,
    k1,
    k2,
    model,
    np,
    r1,
    r2,
    times,
):
    class CompositionalClosedLoopTrajectory(
        closed_loop.ClosedLoopTrajectory
    ):
        """
        Closed-loop rollout on the compositional graph

            V0 --K1--> V1 --K2--> V3
             |          |
             K2         R1 -> done
             |
             v
            V2 --K1--> V3
             |
             R2 -> done

        Assumes target functions are negative inside their sets.
        """

        def __init__(
            self,
            model,
            grid,
            times,
            V0,
            V1,
            V2,
            V3,
            k1,
            k2,
            r1,
            r2,
            capture,
            initial_state,
            steps=160,
            theta=1.0,
            **kwargs,
        ):
            if times[0] >= times[-1]:
                stage_values = {
                    "V0": V0[::-1, ...],
                    "V1": V1[::-1, ...],
                    "V2": V2[::-1, ...],
                    "V3": V3[::-1, ...],
                }
            else:
                raise ValueError(
                    "times must start at terminal time 0 and run backward."
                )

            self._stage_values = stage_values
            self._targets = {
                "K1": k1,
                "K2": k2,
                "R1": r1,
                "R2": r2,
                "capture": capture,
            }
            self._theta = theta

            self._node = "V0"
            self._status = "running"
            self._switch_log = []

            self._nodes = [None] * max(int(steps), 1)
            self._statuses = [None] * max(int(steps), 1)

            # The parent initializes interpolation machinery and calls our
            # overridden _solve_ivp method.
            super().__init__(
                model,
                grid,
                times,
                V0,
                V0,
                initial_state=initial_state,
                steps=steps,
                **kwargs,
            )

        @property
        def switches(self):
            return self._switch_log

        def node(self, t):
            return self._nodes[self._get_sol_index(t)]

        def status(self, t):
            return self._statuses[self._get_sol_index(t)]

        def _target_value(self, name, state):
            value = self._grid.interpolate(
                self._targets[name],
                state=state,
            )
            return float(np.asarray(value))

        def _continuation_value(self, node, t, state):
            value = self._value(
                t,
                state,
                self._stage_values[node],
            )
            return float(np.asarray(value))

        def _record_event(self, t, state, old, new, reason):
            self._switch_log.append(
                {
                    "time": float(t),
                    "remaining_time": float(-t),
                    "from": old,
                    "to": new,
                    "reason": reason,
                    "state": np.asarray(state).copy(),
                }
            )

        def _switch_to(self, new_node, t, state, reason):
            old_node = self._node
            self._node = new_node
            self._V = self._stage_values[new_node]
            self._record_event(
                t, state, old_node, new_node, reason
            )

        def _finish(self, status, t, state, reason):
            self._status = status
            self._record_event(
                t, state, self._node, status, reason
            )

        def _update_task_node(self, t, state):
            if self._status != "running":
                return

            # Capture is positive inside the unsafe set.
            if self._target_value("capture", state) >= 0.0:
                self._finish(
                    "captured",
                    t,
                    state,
                    "capture >= 0",
                )
                return

            if self._node == "V0":
                k1_value = self._target_value("K1", state)
                k2_value = self._target_value("K2", state)

                V1_value = self._continuation_value(
                    "V1", t, state
                )
                V2_value = self._continuation_value(
                    "V2", t, state
                )

                # These mirror ClosedLoopTrajectoryRR.
                if k1_value <= self._theta * V1_value:
                    self._switch_to(
                        "V1",
                        t,
                        state,
                        "K1 <= theta * V1",
                    )
                elif k2_value <= self._theta * V2_value:
                    self._switch_to(
                        "V2",
                        t,
                        state,
                        "K2 <= theta * V2",
                    )

            elif self._node == "V1":
                if self._target_value("R1", state) <= 0.0:
                    self._finish(
                        "completed_R1",
                        t,
                        state,
                        "R1 reached",
                    )
                    return

                k2_value = self._target_value("K2", state)
                V3_value = self._continuation_value(
                    "V3", t, state
                )

                if k2_value <= self._theta * V3_value:
                    self._switch_to(
                        "V3",
                        t,
                        state,
                        "K2 <= theta * V3",
                    )

            elif self._node == "V2":
                if self._target_value("R2", state) <= 0.0:
                    self._finish(
                        "completed_R2",
                        t,
                        state,
                        "R2 reached",
                    )
                    return

                k1_value = self._target_value("K1", state)
                V3_value = self._continuation_value(
                    "V3", t, state
                )

                if k1_value <= self._theta * V3_value:
                    self._switch_to(
                        "V3",
                        t,
                        state,
                        "K1 <= theta * V3",
                    )

            elif self._node == "V3":
                r1_value = self._target_value("R1", state)
                r2_value = self._target_value("R2", state)

                if min(r1_value, r2_value) <= 0.0:
                    reached = (
                        "R1" if r1_value <= r2_value else "R2"
                    )
                    self._finish(
                        f"completed_{reached}",
                        t,
                        state,
                        f"{reached} reached from V3",
                    )

        def _solve_ivp(self, **kwargs):
            import scipy.integrate as _integrate

            state = self._initial_state.copy()

            for i in range(self._steps):
                t = (
                    self._times[-1] - self._times[0]
                ) * (i / self._steps) + self._times[0]

                t_plus = (
                    self._times[-1] - self._times[0]
                ) * ((i + 1) / self._steps) + self._times[0]

                self._check_state_in_bounds(
                    t, state, "pre-solve"
                )
                self._update_task_node(t, state)

                self._nodes[i] = self._node
                self._statuses[i] = self._status

                if self._status == "running":
                    gradient = self._gradient(t, state)
                    self._u = self._model.optimal_control(
                        state, t, gradient
                    )
                    self._d = self._model.optimal_disturbance(
                        state, t, gradient
                    )
                else:
                    # Freeze the terminal state so the remaining animation
                    # frames are still well defined.
                    zero_gradient = np.zeros_like(state)
                    self._u = np.zeros_like(
                        np.asarray(
                            self._model.optimal_control(
                                state, t, zero_gradient
                            )
                        )
                    )
                    self._d = np.zeros_like(
                        np.asarray(
                            self._model.optimal_disturbance(
                                state, t, zero_gradient
                            )
                        )
                    )

                def _dynamics(time, current_state):
                    return self._model(
                        current_state,
                        self._u,
                        self._d,
                        time,
                    )

                solution = _integrate.solve_ivp(
                    _dynamics,
                    [t, t_plus],
                    state,
                    dense_output=True,
                    **kwargs,
                )

                state = solution.sol(t_plus)
                self._check_state_in_bounds(
                    t_plus, state, "post-solve"
                )

                self._sols[i] = solution.sol
                self._us[i] = self._u
                self._ds[i] = self._d


    rollout = CompositionalClosedLoopTrajectory(
        model=model,
        grid=grid,
        times=times,
        V0=V0,
        V1=V1,
        V2=V2,
        V3=V3,
        k1=k1,
        k2=k2,
        r1=r1,
        r2=r2,
        capture=capture,
        initial_state=initial_state,
        steps=160,
        theta=1.0,
        rtol=1e-6,
        atol=1e-8,
    )

    rollout.switches
    return (rollout,)


@app.cell
def _(
    add_emoji_marker,
    decorate_arena,
    horizon,
    np,
    p1_marker_image,
    p2_marker_image,
    plt,
    rollout,
):
    rollout_times = np.linspace(-horizon, 0.0, 400)
    rollout_states = np.stack([rollout.x(t) for t in rollout_times])

    fig_rollout, ax_rollout = plt.subplots(figsize=(7, 7))
    decorate_arena(ax_rollout, show_players=False)
    ax_rollout.plot(
        rollout_states[:, 0],
        rollout_states[:, 1],
        color="purple",
        linewidth=2.5,
        label="P1 optimal path",
    )
    ax_rollout.plot(
        rollout_states[:, 2],
        rollout_states[:, 3],
        color="darkgreen",
        linewidth=2.5,
        label="P2 worst-case path",
    )
    add_emoji_marker(ax_rollout, rollout_states[0, :2], p1_marker_image)
    add_emoji_marker(ax_rollout, rollout_states[0, 2:], p2_marker_image)
    ax_rollout.scatter(
        rollout_states[-1, 0], rollout_states[-1, 1], color="purple", marker="x", s=80
    )
    ax_rollout.scatter(
        rollout_states[-1, 2],
        rollout_states[-1, 3],
        color="darkgreen",
        marker="x",
        s=80,
    )
    ax_rollout.set_title(r"Optimal Play: Rollout from $(-3,0,3,0)$")
    ax_rollout.legend(loc="lower center")
    fig_rollout.tight_layout()
    fig_rollout
    return


@app.cell
def _(
    V0,
    V1,
    V2,
    V3,
    add_emoji_marker,
    decorate_arena,
    grid,
    horizon,
    mo,
    np,
    p1_marker_image,
    p2_marker_image,
    plt,
    rollout,
    times,
):
    from matplotlib.animation import (
        FuncAnimation as _FuncAnimation,
        PillowWriter as _PillowWriter,
    )

    gif_frame_count = 121
    gif_fps = 15
    gif_times = np.linspace(-horizon, 0.0, gif_frame_count)
    gif_states = np.stack(
        [rollout.x(t) for t in gif_times]
    )

    gif_value_functions = {
        "V0": V0,
        "V1": V1,
        "V2": V2,
        "V3": V3,
    }

    gif_x_coordinates = np.asarray(
        grid.coordinate_vectors[0]
    )
    gif_y_coordinates = np.asarray(
        grid.coordinate_vectors[1]
    )
    gif_X, gif_Y = np.meshgrid(
        gif_x_coordinates,
        gif_y_coordinates,
        indexing="ij",
    )

    gif_figure, gif_axis = plt.subplots(figsize=(7, 7), dpi=600)

    gif_time_coordinates = np.asarray(times, dtype=float)
    gif_p2_x_coordinates = np.asarray(
        grid.coordinate_vectors[2],
        dtype=float,
    )
    gif_p2_y_coordinates = np.asarray(
        grid.coordinate_vectors[3],
        dtype=float,
    )


    def _interpolation_bracket(coordinates, query):
        """Return original-array indices bracketing query and its interpolation weight."""
        coordinates = np.asarray(coordinates, dtype=float)
        order = np.argsort(coordinates)
        sorted_coordinates = coordinates[order]

        upper = np.searchsorted(
            sorted_coordinates,
            query,
            side="right",
        )
        upper = int(
            np.clip(upper, 1, len(sorted_coordinates) - 1)
        )
        lower = upper - 1

        lower_coordinate = sorted_coordinates[lower]
        upper_coordinate = sorted_coordinates[upper]

        if np.isclose(lower_coordinate, upper_coordinate):
            weight = 0.0
        else:
            weight = (
                (query - lower_coordinate)
                / (upper_coordinate - lower_coordinate)
            )
            weight = float(np.clip(weight, 0.0, 1.0))

        return int(order[lower]), int(order[upper]), weight


    def _interpolated_level_set_slice(values, t, p2_state):
        """Interpolate V(t, x1, y1, x2, y2) over t, x2, and y2."""
        t0, t1, wt = _interpolation_bracket(
            gif_time_coordinates,
            t,
        )
        x0, x1, wx = _interpolation_bracket(
            gif_p2_x_coordinates,
            p2_state[0],
        )
        y0, y1, wy = _interpolation_bracket(
            gif_p2_y_coordinates,
            p2_state[1],
        )

        field = np.zeros(
            (len(gif_x_coordinates), len(gif_y_coordinates)),
            dtype=float,
        )

        for time_index, time_weight in (
            (t0, 1.0 - wt),
            (t1, wt),
        ):
            for p2_x_index, p2_x_weight in (
                (x0, 1.0 - wx),
                (x1, wx),
            ):
                for p2_y_index, p2_y_weight in (
                    (y0, 1.0 - wy),
                    (y1, wy),
                ):
                    weight = (
                        time_weight
                        * p2_x_weight
                        * p2_y_weight
                    )

                    field += weight * np.asarray(
                        values[
                            time_index,
                            :,
                            :,
                            p2_x_index,
                            p2_y_index,
                        ]
                    )

        return field

    def _draw_gif_frame(frame_index):
        gif_axis.clear()

        t = gif_times[frame_index]
        state = gif_states[frame_index]
        node = rollout.node(t)
    
        status = [
            "running",
            "has key 1",
            "has key 2",
            "has both keys",
        ][int(node[-1])]
    
        if rollout.status(t)[0] == "complete":
            status = "escaped"

        values = gif_value_functions[node]

        field = _interpolated_level_set_slice(
            values,
            t,
            state[2:4],
        )

        # Shade the current negative sublevel set.
        field_min = float(np.nanmin(field))
        field_max = float(np.nanmax(field))

        if field_min < 0.0:
            gif_axis.contourf(
                gif_X,
                gif_Y,
                field,
                levels=[field_min - 1e-9, 0.0],
                colors=["#bed6ab"],
                alpha=0.3,
                zorder=0
            )

        if field_min <= 0.0 <= field_max:
            gif_axis.contour(
                gif_X,
                gif_Y,
                field,
                levels=[0.0],
                colors="black",
                linewidths=2.0,
                # alpha=0.3,
                zorder=0
            )

        decorate_arena(
            gif_axis,
            show_players=False,
        )

        # Paths traversed up through the present frame.
        gif_axis.plot(
            gif_states[: frame_index + 1, 0],
            gif_states[: frame_index + 1, 1],
            color="purple",
            linewidth=2.5,
            label="P1",
            zorder=7,
        )
        gif_axis.plot(
            gif_states[: frame_index + 1, 2],
            gif_states[: frame_index + 1, 3],
            color="darkgreen",
            linewidth=2.5,
            label="P2",
            zorder=7,
        )

        add_emoji_marker(
            gif_axis,
            state[:2],
            p1_marker_image,
            zoom=0.42,
        )
        add_emoji_marker(
            gif_axis,
            state[2:],
            p2_marker_image,
            zoom=0.42,
        )

        remaining_time = max(0.0, -float(t))

        tasks = [
            r"(\mathsf{F}(K_1 \wedge \mathsf{F}R_1) \vee "
            r"\mathsf{F}(K_2 \wedge \mathsf{F}R_2)) \wedge \mathsf{G}\neg C",
    
            r"(\mathsf{F}R_1 \vee "
            r"\mathsf{F}(K_2 \wedge \mathsf{F}R_2)) \wedge \mathsf{G}\neg C",
    
            r"(\mathsf{F}R_2 \vee "
            r"\mathsf{F}(K_1 \wedge \mathsf{F}R_1)) \wedge \mathsf{G}\neg C",
    
            r"(\mathsf{F}R_1 \vee \mathsf{F}R_2) \wedge \mathsf{G}\neg C",
        ]
        task = tasks[int(node[-1])]

        gif_axis.set_title(
            rf"Game: ${tasks[0]}$"
            f"\ntime left: {remaining_time:.1f}" # + " | " + rf"to-go: ${task}$"
        )

        gif_axis.text(
            0.02,
            0.98,
            f"status: {status}",
            transform=gif_axis.transAxes,
            ha="left",
            va="top",
            bbox={
                "facecolor": "white",
                "alpha": 0.85,
                "edgecolor": "none",
            },
            zorder=20,
        )

        completed_switches = [
            event
            for event in rollout.switches
            if event["time"] <= t + 1e-10
        ]
        if completed_switches:
            switch_text = "\n".join(
                f"{event['from']} → {event['to']} "
                f"at τ={event['remaining_time']:.2f}"
                for event in completed_switches
            )
            gif_axis.text(
                0.02,
                0.02,
                switch_text,
                transform=gif_axis.transAxes,
                ha="left",
                va="bottom",
                fontsize=9,
                bbox={
                    "facecolor": "white",
                    "alpha": 0.85,
                    "edgecolor": "none",
                },
                zorder=20,
            )

        gif_axis.legend(loc="lower center")
        return (gif_axis,)


    rollout_animation = _FuncAnimation(
        gif_figure,
        _draw_gif_frame,
        frames=gif_frame_count,
        interval=1000 / gif_fps,
        blit=False,
    )

    animation_path = "twogoal_compositional_rollout.gif"
    rollout_animation.save(
        animation_path,
        writer=_PillowWriter(fps=gif_fps),
        dpi=100,
    )
    plt.close(gif_figure)

    mo.image(
        animation_path,
        width=700,
        caption="Compositional optimal rollout",
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
