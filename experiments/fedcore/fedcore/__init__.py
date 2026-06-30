"""fedcore -- importable package for the Fed-CORE certification core.

Structure-only refactor of the previously-flat experiments/fedcore modules. Old flat module
paths remain as backward-compat shims that re-export from here (shims are leaves; fedcore.*
never imports a flat shim path).
"""
