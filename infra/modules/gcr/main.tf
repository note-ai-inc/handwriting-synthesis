terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

# Create Artifact Registry repositories in each region
resource "google_artifact_registry_repository" "repositories" {
  for_each = toset(var.regions)

  location      = each.value
  repository_id = "handwriting-synthesis-service-images"
  description   = "Container registry for handwriting synthesis service in ${each.value}"
  format        = "DOCKER"
}