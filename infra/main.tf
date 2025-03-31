terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
}

# Variables
variable "project_id" {
  description = "The GCP project ID"
  type        = string
}

variable "service_account_id" {
  description = "The service account ID (project number)"
  type        = string
  default     = "334167120222"  # Your project number
}

variable "repository_url" {
  description = "The Git repository URL"
  type        = string
  default     = "https://github.com/your-username/handwriting-synthesis.git"  # Replace with your actual repo URL
}

variable "region" {
  description = "The default GCP region"
  type        = string
  default     = "asia-northeast3"
}

# Read the SSH keys from files
locals {
  ssh_public_key  = file("${path.module}/../deploy_key.pub")
  ssh_private_key = file("${path.module}/../deploy_key")
}

# VM Module
module "vm" {
  source = "./modules/vm"
  
  project_id        = var.project_id
  service_name      = "handwriting-synthesis"
  service_account_id = var.service_account_id
  region           = var.region
  regions          = ["asia-northeast3", "us-west2", "us-east4"]
  ssh_public_key   = local.ssh_public_key
  ssh_private_key  = local.ssh_private_key
  repository_url   = var.repository_url
}

# VM-LB Module
module "vm-lb" {
  source = "./modules/vm-lb"
  
  project_id    = var.project_id
  service_name  = "handwriting-synthesis"
  domain_name   = "handwriting-synthesis.com"
  vm_self_links = module.vm.vm_self_links
} 