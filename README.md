# Reach / always-avoid game demos

## Environment setup

From the repository root, use the locked project environment:

```bash
uv sync
uv run marimo edit hjr_twogoal_PEgame.py
```

The two-goal notebook renders markers with macOS's native Apple Color Emoji.
The Python module `AppKit` used for that renderer is provided by
`pyobjc-framework-Cocoa`, which is already declared in `pyproject.toml` and
`uv.lock`.

If installing into an existing macOS environment without `uv`, use:

```bash
python -m pip install emoji pillow pyobjc-framework-Cocoa
```

Do **not** install the PyPI distribution named `AppKit`. It is an unrelated
Flask/GTK project that attempts to build PyGObject and Cairo.

If that package was installed accidentally, repair the environment with:

```bash
python -m pip uninstall -y AppKit
python -m pip install --upgrade emoji pillow pyobjc-framework-Cocoa
```
