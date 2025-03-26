terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

# Regional provider for GCR resources
provider "google" {
  project = var.project_id
  region  = var.region
}

module "gcr" {
  source = "../../modules/gcr"

  project_id = var.project_id
  regions    = var.regions
}

module "global_lb" {
  source = "../../modules/global-lb"

  project_id    = var.project_id
  regions       = var.regions
  service_name  = "handwriting-synthesis"
  domain_name   = var.domain_name
}
