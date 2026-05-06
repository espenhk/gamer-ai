# ============================================================================
# variables.tf
# ============================================================================

variable "location" {
  description = "Azure region for all resources"
  type        = string
  default     = "centralindia"
}

variable "project_name" {
  description = "Short project name used in resource naming"
  type        = string
  default     = "game"
}

variable "worker_vm_count" {
  description = "Number of worker VMs to create"
  type        = number
  default     = 2
}

variable "worker_vm_size" {
  description = "Azure VM size for worker VMs"
  type        = string
  default     = "Standard_D2as_v5"
}

variable "coordinator_vm_size" {
  description = "Azure VM size for the coordinator VM"
  type        = string
  default     = "Standard_B1ms"
}

variable "admin_username" {
  description = "Local admin username for the VMs"
  type        = string
  default     = "adminuser"
}

variable "my_object_id" {
  description = "Your Azure AD user object ID. Find with: az ad signed-in-user show --query id -o tsv"
  type        = string
}

variable "my_ip_address" {
  description = "Your public IP for RDP access (e.g. 203.0.113.10)"
  type        = string
}

variable "repo_url" {
  description = "Git repo to clone onto worker VMs"
  type        = string
  default     = "https://github.com/espenhk/tmnf-ai.git"
}

variable "worker_game" {
  description = "The game each worker VM should run. Must be one of: tmnf, sc2, torcs, beamng, car_racing. Controls which runtime services setup_and_run.ps1 starts on boot, and which work items the distributed worker accepts."
  type        = string
  default     = "tmnf"

  validation {
    condition     = contains(["tmnf", "sc2", "torcs", "beamng", "car_racing"], var.worker_game)
    error_message = "worker_game must be one of: tmnf, sc2, torcs, beamng, car_racing."
  }
}

variable "worker_command" {
  description = "Python command to run on each worker VM after game services start. Leave empty to only perform setup (no training). Example: \"python -m distributed.worker --coordinator http://10.0.0.5:5555 --token mytoken --game tmnf --no-interrupt\""
  type        = string
  default     = ""
}

variable "coordinator_ip" {
  description = "Private IP address of the coordinator VM. Used to build the default worker command when worker_command is not explicitly set. Ignored if worker_command is provided."
  type        = string
  default     = ""
}

variable "grid_token" {
  description = "Shared secret token for coordinator/worker authentication. Must match the token used when starting the coordinator (python grid_search.py ... --distribute). Set via TMNF_GRID_TOKEN env var on the coordinator, or pass here to bake it into the worker startup command."
  type        = string
  default     = ""
  sensitive   = true
}