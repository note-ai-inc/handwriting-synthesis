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

# Read the SSH keys from files
locals {
  ssh_public_key  = file("${path.root}/../../deploy_key.pub")
  ssh_private_key = file("${path.root}/../../deploy_key")
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

module "vm" {
  source = "../../modules/vm"
  
  project_id        = var.project_id
  regions          = var.regions
  service_name      = "handwriting-synthesis-vm"
  service_account_id = var.service_account_id
  region           = var.region
  ssh_public_key    = local.ssh_public_key
  ssh_private_key   = local.ssh_private_key
  repository_url    = var.repository_url
}

module "vm_lb" {
  source = "../../modules/vm-lb"
  
  project_id    = var.project_id
  service_name  = "handwriting-synthesis-vm"
  domain_name   = var.vm_domain_name
  regions       = var.regions
  vm_self_links = module.vm.vm_self_links
  
  depends_on = [module.vm]
}

module "iam" {
  source = "../../modules/iam"

  project_id = var.project_id
}
