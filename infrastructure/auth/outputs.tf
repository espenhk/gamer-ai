output "service_principal_application_id" {
  description = "Application ID of the service principal"
  value       = azuread_application.main.client_id
}

output "service_principal_object_id" {
  description = "Object ID of the service principal"
  value       = azuread_service_principal.main.object_id
}

output "service_principal_password" {
  description = "Password for the service principal"
  value       = azuread_service_principal_password.main.value
  sensitive   = true
}

output "service_principal_credentials" {
  description = "Service principal credentials for use in CI/CD"
  value = {
    client_id       = azuread_application.main.client_id
    client_secret   = azuread_service_principal_password.main.value
    tenant_id       = data.azuread_client_config.current.tenant_id
    subscription_id = data.azurerm_client_config.current.subscription_id
  }
  sensitive = true
}
