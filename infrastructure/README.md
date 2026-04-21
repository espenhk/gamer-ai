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
