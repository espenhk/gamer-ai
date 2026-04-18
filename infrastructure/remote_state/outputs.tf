output "storage_account_name" {
  description = "Name of the storage account for remote state"
  value       = azurerm_storage_account.tfstate.name
}

output "storage_account_id" {
  description = "ID of the storage account for remote state"
  value       = azurerm_storage_account.tfstate.id
}

output "resource_group_name" {
  description = "Name of the resource group for remote state"
  value       = azurerm_resource_group.tfstate.name
}

output "resource_group_id" {
  description = "ID of the resource group for remote state"
  value       = azurerm_resource_group.tfstate.id
}

output "container_name" {
  description = "Names of the storage container"
  value       = azurerm_storage_container.tfstate.name
}

output "backend_config" {
  description = "Backend configuration for each environment"
  value = {
    resource_group_name  = azurerm_resource_group.tfstate.name
    storage_account_name = azurerm_storage_account.tfstate.name
    container_name       = azurerm_storage_container.tfstate.name
    key                  = "terraform.tfstate"
  }
}
