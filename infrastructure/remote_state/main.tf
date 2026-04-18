# Random string for unique naming
resource "random_string" "suffix" {
  length  = 8
  special = false
  upper   = false
}

# Resource Group for remote state
resource "azurerm_resource_group" "tfstate" {
  name     = "rg-tfstate-${var.project_name}"
  location = var.location

  tags = {
    Purpose     = "Terraform Remote State"
    Project     = var.project_name
    ManagedBy   = "Terraform"
    Environment = "shared"
  }
}

# Storage Account for remote state
resource "azurerm_storage_account" "tfstate" {
  name                     = "sttfstate${random_string.suffix.result}"
  resource_group_name      = azurerm_resource_group.tfstate.name
  location                 = azurerm_resource_group.tfstate.location
  account_tier             = var.storage_account_tier
  account_replication_type = var.storage_replication_type
  min_tls_version          = "TLS1_2"

  # Disable public blob access
  allow_nested_items_to_be_public = false

  # Enable infrastructure encryption
  infrastructure_encryption_enabled = true

  public_network_access_enabled = true  # needed so your CI/local machine can reach it

  network_rules {
    default_action = "Deny"
    ip_rules       = [var.my_ip_address]  # your public IP (or CI runner IP)
    bypass         = ["AzureServices"]
  }

  blob_properties {
    versioning_enabled = var.enable_versioning

    dynamic "delete_retention_policy" {
      for_each = var.enable_soft_delete ? [1] : []
      content {
        days = var.soft_delete_retention_days
      }
    }

    dynamic "container_delete_retention_policy" {
      for_each = var.enable_soft_delete ? [1] : []
      content {
        days = var.soft_delete_retention_days
      }
    }
  }

  tags = {
    Purpose     = "Terraform Remote State"
    Project     = var.project_name
    ManagedBy   = "Terraform"
    Environment = "shared"
  }
}

# Single container for all environments
resource "azurerm_storage_container" "tfstate" {
  storage_account_name = azurerm_storage_account.tfstate.name
  name = "tfstate"
  container_access_type = "private"

}

# Management policy for lifecycle management
resource "azurerm_storage_management_policy" "tfstate" {
  storage_account_id = azurerm_storage_account.tfstate.id

  rule {
    name    = "tfstate-lifecycle"
    enabled = true
    filters {
      prefix_match = ["tfstate"]
      blob_types   = ["blockBlob"]
    }

    actions {
      version {
        delete_after_days_since_creation = 90
      }
    }
  }
}

# Lock storage account to prevent accidental deletion
#resource "azurerm_management_lock" "tfstate_storage" {
#  name       = "tfstate-storage-lock"
#  scope      = azurerm_storage_account.tfstate.id
#  lock_level = "CanNotDelete"
#  notes      = "Prevents accidental deletion of Terraform state storage"
#} # TODO this needs to be put in place
