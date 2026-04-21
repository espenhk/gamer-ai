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