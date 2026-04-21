# ============================================================================
# data sources
# ============================================================================

data "azurerm_client_config" "current" {}


# tags

locals {
  common_tags = {
    ghs-dataclassification = "Public"
    ghs-deployedby         = "Manual"
    ghs-environmenttype    = "Sandbox"
    ghs-los                = "Advisory"
    ghs-serviceoffering    = "Sandbox"
    ghs-solution           = "PwC Alliance Sandbox"
    ghs-solutionexposure   = "PwC Internal"
    ghs-tariff             = "zab"
  }

}

# ============================================================================
# Resource Group
# ============================================================================

resource "azurerm_resource_group" "main" {
  name     = "rg-${var.project_name}"
  location = var.location

  tags = local.common_tags
}

# ============================================================================
# Networking — minimal VNet + subnet (required by Azure), zero cost
# ============================================================================

resource "azurerm_virtual_network" "main" {
  name                = "vnet-${var.project_name}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  address_space       = ["10.0.0.0/24"]

  tags = local.common_tags
}

resource "azurerm_subnet" "vms" {
  name                 = "snet-vms"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = ["10.0.0.0/24"]
}

# ============================================================================
# NSG — RDP inbound from your IP only, all outbound allowed by default
# ============================================================================

resource "azurerm_network_security_group" "vms" {
  name                = "nsg-${var.project_name}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name

  security_rule {
    name                       = "AllowRDP"
    priority                   = 100
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "3389"
    source_address_prefix      = var.my_ip_address
    destination_address_prefix = "*"
  }

  # Azure NSGs have a built-in AllowInternetOutbound rule at priority 65001.
  # VMs can freely download from the internet without any explicit outbound rule.

  tags = local.common_tags
}

resource "azurerm_subnet_network_security_group_association" "vms" {
  subnet_id                 = azurerm_subnet.vms.id
  network_security_group_id = azurerm_network_security_group.vms.id
}

# ============================================================================
# Coordinator — Public IP + NIC
# ============================================================================

resource "azurerm_public_ip" "coordinator" {
  name                = "pip-${var.project_name}-coordinator"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  allocation_method   = "Static"
  sku                 = "Standard"

  tags = local.common_tags
}

resource "azurerm_network_interface" "coordinator" {
  name                = "nic-${var.project_name}-coordinator"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name

  ip_configuration {
    name                          = "primary"
    subnet_id                     = azurerm_subnet.vms.id
    public_ip_address_id          = azurerm_public_ip.coordinator.id
    private_ip_address_allocation = "Dynamic"
  }

  tags = local.common_tags
}

# ============================================================================
#  Workers — Public IPs + NICs (one per VM)
# ============================================================================

resource "azurerm_public_ip" "worker" {
  count               = var.worker_vm_count
  name                = "pip-${var.project_name}-${count.index}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  allocation_method   = "Static"
  sku                 = "Standard"

  tags = local.common_tags
}

resource "azurerm_network_interface" "worker" {
  count               = var.worker_vm_count
  name                = "nic-${var.project_name}-${count.index}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name

  ip_configuration {
    name                          = "primary"
    subnet_id                     = azurerm_subnet.vms.id
    public_ip_address_id          = azurerm_public_ip.worker[count.index].id
    private_ip_address_allocation = "Dynamic"
  }

  tags = local.common_tags
}

# ============================================================================
# Key Vault — no inline access policies
# ============================================================================

resource "random_string" "kv_suffix" {
  length  = 6
  lower   = true
  upper   = false
  numeric = true
  special = false
}

resource "azurerm_key_vault" "main" {
  name                       = "kv-${var.project_name}-${random_string.kv_suffix.result}"
  location                   = azurerm_resource_group.main.location
  resource_group_name        = azurerm_resource_group.main.name
  tenant_id                  = data.azurerm_client_config.current.tenant_id
  sku_name                   = "standard"
  soft_delete_retention_days = 7
  purge_protection_enabled   = false
  enabled_for_deployment     = true

  tags = local.common_tags
}

# ============================================================================
# Key Vault Access Policies — all separate resources
# ============================================================================

# Terraform SP — full secret management for apply/destroy
resource "azurerm_key_vault_access_policy" "terraform_sp" {
  key_vault_id = azurerm_key_vault.main.id
  tenant_id    = data.azurerm_client_config.current.tenant_id
  object_id    = data.azurerm_client_config.current.object_id

  secret_permissions = [
    "Backup", "Delete", "Get", "List", "Purge", "Recover", "Restore", "Set",
  ]
}

resource "azurerm_key_vault_access_policy" "coordinator" {
  key_vault_id = azurerm_key_vault.main.id
  tenant_id    = data.azurerm_client_config.current.tenant_id
  object_id    = azurerm_windows_virtual_machine.coordinator.identity[0].principal_id

  secret_permissions = [
    "Get",
  ]
}

resource "azurerm_key_vault_access_policy" "worker" {
  count        = var.worker_vm_count
  key_vault_id = azurerm_key_vault.main.id
  tenant_id    = data.azurerm_client_config.current.tenant_id
  object_id    = azurerm_windows_virtual_machine.worker[count.index].identity[0].principal_id

  secret_permissions = [
    "Get",
  ]
}
# ============================================================================
# Passwords — generated and stored in Key Vault
# ============================================================================

resource "random_password" "coordinator" {
  length           = 24
  special          = true
  override_special = "!@#$%&*()-_=+"
  min_upper        = 2
  min_lower        = 2
  min_numeric      = 2
  min_special      = 2
}

resource "random_password" "worker" {
  count            = var.worker_vm_count
  length           = 24
  special          = true
  override_special = "!@#$%&*()-_=+"
  min_upper        = 2
  min_lower        = 2
  min_numeric      = 2
  min_special      = 2
}

resource "azurerm_key_vault_secret" "coordinator_password" {
  name         = "${var.project_name}-coordinator-password"
  value        = random_password.coordinator.result
  key_vault_id = azurerm_key_vault.main.id

  depends_on = [azurerm_key_vault_access_policy.terraform_sp]
}

resource "azurerm_key_vault_secret" "worker_password" {
  count        = var.worker_vm_count
  name         = "${var.project_name}-worker-${count.index}-password"
  value        = random_password.worker[count.index].result
  key_vault_id = azurerm_key_vault.main.id

  depends_on = [azurerm_key_vault_access_policy.terraform_sp]
}

# ============================================================================
# Windows VMs
# ============================================================================

resource "azurerm_windows_virtual_machine" "coordinator" {
  name                = "vm-${var.project_name}-coord"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  size                = var.coordinator_vm_size
  admin_username      = var.admin_username
  admin_password      = random_password.coordinator.result

  network_interface_ids = [azurerm_network_interface.coordinator.id]

  identity {
    type = "SystemAssigned"
  }

  os_disk {
    name                 = "osdisk-vm-${var.project_name}-coordinator"
    caching              = "ReadWrite"
    storage_account_type = "Standard_LRS"
  }

  source_image_reference {
    publisher = "MicrosoftWindowsDesktop"
    offer     = "windows-11"
    sku       = "win11-24h2-pro"
    version   = "latest"
  }

  tags = local.common_tags
}

resource "azurerm_virtual_machine_extension" "clone_repo_to_coordinator" {
  name                 = "clone-repo"
  virtual_machine_id   = azurerm_windows_virtual_machine.coordinator.id
  publisher            = "Microsoft.Compute"
  type                 = "CustomScriptExtension"
  type_handler_version = "1.10"

  protected_settings = jsonencode({
    commandToExecute = "powershell -ExecutionPolicy Unrestricted -Command \"[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://github.com/git-for-windows/git/releases/download/v2.47.1.windows.2/Git-2.47.1.2-64-bit.exe' -OutFile 'C:\\git-installer.exe' -UseBasicParsing; Start-Process -FilePath 'C:\\git-installer.exe' -ArgumentList '/VERYSILENT /NORESTART /NOCANCEL /SP- /CLOSEAPPLICATIONS /RESTARTAPPLICATIONS /COMPONENTS=\\\"icons,ext\\\\shellhere,ext\\\\guihere,gitlfs,assoc,assoc_sh\\\"' -Wait; $env:Path = [System.Environment]::GetEnvironmentVariable('Path','Machine'); git clone ${var.repo_url} C:\\tmnf-ai\""
  })

  tags = local.common_tags
}

resource "azurerm_windows_virtual_machine" "worker" {
  count               = var.worker_vm_count
  name                = "vm-${var.project_name}-${count.index}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  size                = var.worker_vm_size
  admin_username      = var.admin_username
  admin_password      = random_password.worker[count.index].result

  network_interface_ids = [azurerm_network_interface.worker[count.index].id]

  identity {
    type = "SystemAssigned"
  }

  os_disk {
    name                 = "osdisk-vm-worker-${var.project_name}-${count.index}"
    caching              = "ReadWrite"
    storage_account_type = "Standard_LRS"
  }

  source_image_reference {
    publisher = "MicrosoftWindowsDesktop"
    offer     = "windows-11"
    sku       = "win11-24h2-pro"
    version   = "latest"
  }

  tags = local.common_tags
}

resource "azurerm_virtual_machine_extension" "setup" {
  count                = var.worker_vm_count
  name                 = "setup"
  virtual_machine_id   = azurerm_windows_virtual_machine.worker[count.index].id
  publisher            = "Microsoft.Compute"
  type                 = "CustomScriptExtension"
  type_handler_version = "1.10"

  protected_settings = jsonencode({
    commandToExecute = join("; ", [
      # Install Git silently
      "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12",
      "Invoke-WebRequest -Uri 'https://github.com/git-for-windows/git/releases/download/v2.47.1.windows.2/Git-2.47.1.2-64-bit.exe' -OutFile C:\\git-installer.exe -UseBasicParsing",
      "Start-Process -FilePath C:\\git-installer.exe -ArgumentList '/VERYSILENT','/NORESTART' -Wait",
      "$env:Path = [System.Environment]::GetEnvironmentVariable('Path','Machine')",

      # Clone repo (skip if exists)
      "if (!(Test-Path C:\\tmnf-ai)) { & git clone ${var.repo_url} C:\\tmnf-ai }",

      # Register startup task (runs setup_and_run.ps1 at every boot)
      "$a = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument '-ExecutionPolicy Unrestricted -File C:\\tmnf-ai\\setup_and_run.ps1'",
      "$t = New-ScheduledTaskTrigger -AtStartup",
      "$p = New-ScheduledTaskPrincipal -UserId 'SYSTEM' -RunLevel Highest",
      "Register-ScheduledTask -TaskName 'TMNF-AI-Setup' -Action $a -Trigger $t -Principal $p -Force"
    ])

    # This tells Azure to run it as PowerShell, not cmd.exe
    commandToExecute = "powershell.exe -ExecutionPolicy Unrestricted -Command \"${join("; ", [
      "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12",
      "Invoke-WebRequest -Uri 'https://github.com/git-for-windows/git/releases/download/v2.47.1.windows.2/Git-2.47.1.2-64-bit.exe' -OutFile C:\\git-installer.exe -UseBasicParsing",
      "Start-Process -FilePath C:\\git-installer.exe -ArgumentList '/VERYSILENT','/NORESTART' -Wait",
      "$env:Path = [System.Environment]::GetEnvironmentVariable('Path','Machine')",
      "if (!(Test-Path C:\\tmnf-ai)) { & git clone ${var.repo_url} C:\\tmnf-ai }",
      "$a = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument '-ExecutionPolicy Unrestricted -File C:\\tmnf-ai\\setup_and_run.ps1'",
      "$t = New-ScheduledTaskTrigger -AtStartup",
      "$p = New-ScheduledTaskPrincipal -UserId 'SYSTEM' -RunLevel Highest",
      "Register-ScheduledTask -TaskName 'TMNF-AI-Setup' -Action $a -Trigger $t -Principal $p -Force",
    ])}\""
  })

  tags = local.common_tags
}