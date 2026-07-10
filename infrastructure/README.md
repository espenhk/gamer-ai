# Infrastructure code for running Azure VMs as runner infrastructure for the project

## Deployment

First time deploying, deploy resources in the following order:

1. auth: terraform init, plan, apply
2. remote_state: terraform init, plan, apply
3. make a backend-config file based on backend.conf.example, place in environment/, fill in the values for your backend as deployed from remote_state
4. in environment/, run

```sh
 terraform init terraform init -backend-config=backend.conf
```

5. then terraform plan and apply

## Running/stopping VMs

To stop your VMs:

```sh
az vm deallocate --ids $(az vm list -g rg-tmnf --query "[].id" -o tsv)
```

To re-start them:

```sh
az vm start --ids $(az vm list -g rg-tmnf --query "[].id" -o tsv)
```

## Multi-game distributed training

All worker VMs automatically install every supported game on first boot.  To
run a distributed experiment for a specific game:

### 1. Choose your game and token

Edit `terraform.tfvars` (copy from `terraform.tfvars.example`):

```hcl
worker_game    = "tmnf"          # or sc2, torcs, beamng, car_racing
worker_command = ""              # leave empty for initial setup
grid_token     = "mysharedtoken"
coordinator_ip = ""              # fill in after coordinator is deployed
```

### 2. Deploy / re-apply

```sh
terraform apply
```

All worker VMs will install all games on their next boot and start runtime
services for the selected `worker_game`.

### 3. Start the coordinator (on the coordinator VM)

SSH/RDP into the coordinator VM and run:

```sh
cd C:\tmnf-ai
$env:TMNF_GRID_TOKEN = "mysharedtoken"
poetry run python grid_search.py config/my_grid.yaml --game tmnf --distribute
```

Note the coordinator's private IP address (visible in Azure Portal or
`terraform output`).

### 4. Configure workers to connect

Update `terraform.tfvars` with the coordinator's private IP and re-apply:

```hcl
coordinator_ip = "10.0.0.5"
worker_command = "python -m distributed.worker --coordinator http://10.0.0.5:5555 --token mysharedtoken --game tmnf --no-interrupt"
```

```sh
terraform apply
```

Workers will pick up the new startup command on their next boot.  To trigger
an immediate restart:

```sh
az vm restart --ids $(az vm list -g rg-tmnf --query "[].id" -o tsv)
```

### 5. Switching games

To switch from TMNF to SC2 (for example), update `terraform.tfvars`:

```hcl
worker_game    = "sc2"
worker_command = "python -m distributed.worker --coordinator http://10.0.0.5:5555 --token mysharedtoken --game sc2 --no-interrupt"
```

Then `terraform apply` and restart the VMs.  Because all games were installed
on first boot, no additional installation time is needed.

### Manually running setup_and_run.ps1 on a worker

```powershell
# Setup only (all games installed, SC2 runtime services started):
.\setup_and_run.ps1 -Game sc2

# Setup + start a distributed worker for SC2:
.\setup_and_run.ps1 -Game sc2 "python -m distributed.worker --coordinator http://10.0.0.5:5555 --token mytoken --game sc2 --no-interrupt"

# Dry run to inspect what would be installed:
.\setup_and_run.ps1 -Game torcs -DryRun
```

## Single-VM unattended long-horizon run (issue #489)

Some experiments — the TMNF long-horizon SAC run being the first — are a
single multi-day training job, not a grid search. Use one `environment/`
Windows VM directly, **not** the distributed coordinator/worker pool above:
there is no grid to split across workers, so the coordinator/worker HTTP
protocol and `--distribute` machinery in `grid_search.py` would only add
moving parts with nothing to coordinate.

This section documents the intended setup; it is **not** something this
repo's automation provisions or runs on your behalf — renting the VM hours
and kicking off the run are deliberate, human-approved actions (real Azure
spend, a real multi-day job against a live TMNF process).

### 1. Size and provision a single worker VM

In `terraform.tfvars`:

```hcl
worker_vm_count = 1                 # one experiment, not a grid search
worker_game     = "tmnf"
worker_command  = ""                # fill in after step 2 (see step 3)
```

Deploy with the usual `terraform init` / `plan` / `apply` sequence described
above. `worker_vm_count = 1` still deploys through the same `environment/`
module as the coordinator/worker pool — it's the same VM shape, just one of
them, driven directly rather than via the distributed protocol.

### 2. Materialize the experiment directory with the SAC config

RDP into the VM once and run, from the cloned repo (`C:\tmnf-ai` by
default):

```powershell
cd C:\tmnf-ai
poetry install --with deep_rl   # pulls stable-baselines3 + torch, needed for policy_type: sac
poetry run python grid_search.py games/tmnf/config/gs_sac_long_horizon.yaml --game tmnf
```

`gs_sac_long_horizon.yaml` (issues #489, #494) is a single-combo grid-search
template — see the file's header comment for the full rationale — that pins
`n_lidar_rays: 16`, `policy_type: sac` with a starting `policy_params`
budget, `action_window_ticks: 5` (#494's 20 Hz control rate), and
`checkpoint_freq` sized for a multi-day run. This one-time invocation
creates `experiments/tmnf/sac/a03_centerline/gs_sac_long_horizon/` with
those values already written into its `training_params.yaml` /
`reward_config.yaml`, and starts training.

### 3. Hand off to the crash-restart supervisor

Once the experiment directory exists (from step 2, even if that first run
later crashes), point the VM at the supervisor instead of a one-shot
command. Update `terraform.tfvars`:

```hcl
worker_command = "powershell -File C:\\tmnf-ai\\infrastructure\\environment\\run_supervised.ps1 -Experiment gs_sac_long_horizon"
```

`terraform apply` and restart the VM (or run the same command directly over
RDP without waiting for a reboot). `run_supervised.ps1` loops
`main.py gs_sac_long_horizon --game tmnf --no-interrupt`, restarting it with
a backoff delay whenever it exits non-zero. Each restart resumes from the
SB3 checkpoint / replay-buffer files written by the crash-safe resume
mechanism (issue #488) rather than retraining from scratch — see
CLAUDE.md's "Crash-safe resume" section. The supervisor stops on its own
once `main.py` exits cleanly (code 0).

### 4. Monitor

Progress is visible the same way as any other experiment: the `results/`
analytics output under the experiment directory, plus #481's canonical-score
tracking once that lands. No separate dashboard is provisioned for this.
