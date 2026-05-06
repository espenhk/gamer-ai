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
