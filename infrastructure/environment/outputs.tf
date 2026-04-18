# ============================================================================
# outputs.tf
# ============================================================================

output "resource_group_name" {
  value = azurerm_resource_group.main.name
}

output "key_vault_name" {
  value = azurerm_key_vault.main.name
}

output "key_vault_uri" {
  value = azurerm_key_vault.main.vault_uri
}

output "vm_details" {
  description = "VM names, public IPs, and Key Vault secret names"
  value = [
    for i in range(var.worker_vm_count) : {
      vm_name              = azurerm_windows_virtual_machine.worker[i].name
      public_ip            = azurerm_public_ip.worker[i].ip_address
      admin_user           = var.admin_username
      password_secret_name = azurerm_key_vault_secret.worker_password[i].name
    }
  ]
}