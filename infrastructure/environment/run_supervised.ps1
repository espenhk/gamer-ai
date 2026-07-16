<#
.SYNOPSIS
    Crash-restart supervisor for a single long-horizon TMNF training run (issue #489).

.DESCRIPTION
    Repeatedly runs `main.py <Experiment> --game tmnf --no-interrupt` and
    restarts it whenever it exits non-zero (crash, TMInterface disconnect,
    transient game-process death, etc). Each restart resumes from the SB3
    checkpoint/replay-buffer files written by the crash-safe resume mechanism
    (issue #488, PR #488) instead of retraining from scratch — see
    CLAUDE.md's "Crash-safe resume" section for how `total_timesteps()` and
    `checkpoint_freq` make this safe.

    This script assumes the experiment directory already exists with the
    desired training_params.yaml / reward_config.yaml (e.g. materialized once
    via `grid_search.py games/tmnf/config/gs_sac_long_horizon.yaml --game tmnf`,
    or hand-created). It does not create or edit experiment config itself.

    Intended for a single Azure VM provisioned via infrastructure/environment/
    (not the distributed coordinator/worker pool — see infrastructure/README.md's
    "Single-VM unattended long-horizon run" section). Set worker_vm_count = 1
    and point worker_command at this script (see that section for the exact
    Terraform variable values).

.PARAMETER Experiment
    Experiment name, matching the directory under
    experiments/tmnf/<policy>/<track>/<Experiment>/.

.PARAMETER RepoDir
    Path to the cloned repo on the VM (setup_and_run.ps1's default clone
    target). Default: C:\tmnf-ai.

.PARAMETER MaxRestarts
    Give up after this many restarts (0 = unlimited, the default — appropriate
    for a genuinely unattended multi-day run).

.PARAMETER RestartDelaySeconds
    Wait this long before restarting after a crash, so a persistent failure
    (e.g. TMInterface never reconnecting) doesn't spin-loop.

.EXAMPLE
    powershell -File run_supervised.ps1 -Experiment gs_sac_long_horizon

.EXAMPLE
    # As a Terraform worker_command (single quotes shown for clarity; use the
    # exact quoting your shell/terraform.tfvars requires):
    worker_command = "powershell -File C:\tmnf-ai\infrastructure\environment\run_supervised.ps1 -Experiment gs_sac_long_horizon"
#>

param(
    [Parameter(Mandatory = $true)]
    [string]$Experiment,

    [string]$RepoDir = "C:\tmnf-ai",

    [int]$MaxRestarts = 0,

    [int]$RestartDelaySeconds = 30
)

$ErrorActionPreference = "Stop"

function Write-Supervisor {
    param([string]$Message)
    Write-Host "[supervisor $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message"
}

Push-Location $RepoDir
try {
    $attempt = 0
    while ($true) {
        $attempt++
        Write-Supervisor "attempt $attempt : poetry run python main.py $Experiment --game tmnf --no-interrupt"

        & poetry run python main.py $Experiment --game tmnf --no-interrupt
        $exitCode = $LASTEXITCODE

        if ($exitCode -eq 0) {
            Write-Supervisor "main.py exited cleanly (code 0) — run complete, stopping supervisor."
            break
        }

        Write-Supervisor "main.py exited with code $exitCode (crash) — will resume from the last SB3 checkpoint (issue #488) on restart."

        # $attempt counts runs (including the initial one), so restarts
        # performed so far = $attempt - 1. Give up only once that many
        # restarts have already been spent, so MaxRestarts=1 really allows
        # one restart (two run attempts total).
        $restartsSoFar = $attempt - 1
        if ($MaxRestarts -gt 0 -and $restartsSoFar -ge $MaxRestarts) {
            Write-Supervisor "reached MaxRestarts=$MaxRestarts restarts, giving up."
            exit $exitCode
        }

        Write-Supervisor "waiting $RestartDelaySeconds s before restart..."
        Start-Sleep -Seconds $RestartDelaySeconds
    }
} finally {
    Pop-Location
}
