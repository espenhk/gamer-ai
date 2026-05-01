#!/usr/bin/env bash
# POSIX counterpart to launch_assetto.ps1.
#
# Assetto Corsa is Windows-only at the binary level, so this script is a
# no-op on Linux/macOS. It exists so CI workflows can invoke a single
# launcher path on every OS without OS-specific branching.

set -euo pipefail

case "$(uname -s)" in
    Linux*|Darwin*)
        echo "launch_assetto.sh: Assetto Corsa is Windows-only; this is a no-op on $(uname -s)."
        echo "On Linux you can run training against the gym wrapper without the AC binary."
        exit 0
        ;;
    *)
        echo "launch_assetto.sh: unsupported platform $(uname -s)" >&2
        exit 1
        ;;
esac
