data "azurerm_client_config" "current" {}
data "azuread_client_config" "current" {}

# Random password for service principal
resource "random_password" "sp_password" {
  length  = 32
  special = true
}

# Create Azure AD Application
resource "azuread_application" "main" {
  display_name = "sp-${var.project_name}"
  owners       = [data.azuread_client_config.current.object_id]

  required_resource_access {
    resource_app_id = "00000003-0000-0000-c000-000000000000" # Microsoft Graph

    resource_access {
      id   = "e1fe6dd8-ba31-4d61-89e7-88639da4683d" # User.Read
      type = "Scope"
    }

    # Application permissions for managing Entra ID applications
    resource_access {
      id   = "1bfefb4e-e0b5-418b-a88f-73c46d2cc8e9" # Application.ReadWrite.All
      type = "Role"
    }

    resource_access {
      id   = "7ab1d382-f21e-4acd-a863-ba3e13f7da61" # Directory.Read.All
      type = "Role"
    }
  }
}

# Create Service Principal
resource "azuread_service_principal" "main" {
  client_id                    = azuread_application.main.client_id
  app_role_assignment_required = false
  owners                       = [data.azuread_client_config.current.object_id]
}

# Create Service Principal Password
resource "azuread_service_principal_password" "main" {
  service_principal_id = azuread_service_principal.main.id
  display_name         = "terraform-password"
}


# Add federated credentials for GitHub OIDC
# Federated credential for main branch
resource "azuread_application_federated_identity_credential" "github_main" {
  application_id = azuread_application.main.id
  display_name   = "github-actions-main"
  description    = "GitHub Actions - main branch"
  audiences      = ["api://AzureADTokenExchange"]
  issuer         = "https://token.actions.githubusercontent.com"
  subject        = "repo:espenhk/tmnf-ai:ref:refs/heads/main"
}

# Federated credential for pull requests
resource "azuread_application_federated_identity_credential" "github_pr" {
  application_id = azuread_application.main.id
  display_name   = "github-actions-pr"
  description    = "GitHub Actions - Pull Requests"
  audiences      = ["api://AzureADTokenExchange"]
  issuer         = "https://token.actions.githubusercontent.com"
  subject        = "repo:espenhk/tmnf-ai:pull_request"
}

# Federated credential for feature branch
resource "azuread_application_federated_identity_credential" "github_feature" {
  application_id = azuread_application.main.id
  display_name   = "github-actions-feature-deploy"
  description    = "GitHub Actions - feature/deploy-actions branch"
  audiences      = ["api://AzureADTokenExchange"]
  issuer         = "https://token.actions.githubusercontent.com"
  subject        = "repo:espenhk/tmnf-ai:ref:refs/heads/feature/deploy-actions"
}

# Get current subscription
data "azurerm_subscription" "current" {}

# Assign Contributor role to the service principal at subscription level
resource "azurerm_role_assignment" "contributor" {
  scope                = data.azurerm_subscription.current.id
  role_definition_name = "Contributor"
  principal_id         = azuread_service_principal.main.object_id
}

# Assign user access administrator role for creating role assignments
#resource "azurerm_role_assignment" "user_access_admin" {
#  scope                = data.azurerm_subscription.current.id
#  role_definition_name = "User Access Administrator"
#  principal_id         = azuread_service_principal.main.object_id
#}
