variable "project_id" {
  description = "The GCP project ID"
  type        = string
}

variable "region" {
  description = "The GCP region"
  type        = string
}

variable "service_account_id" {
  description = "The service account ID"
  type        = string
}

variable "ssh_public_key" {
  description = "The SSH public key for the deploy user"
  type        = string
}

variable "ssh_private_key" {
  description = "The SSH private key for GitHub access"
  type        = string
  sensitive   = true
}

variable "repository_url" {
  description = "The Git repository URL"
  type        = string
}

variable "service_name" {
  description = "The name of the service"
  type        = string
  default     = "handwriting-synthesis"
}

variable "regions" {
  description = "List of regions to deploy VMs to"
  type        = list(string)
  default     = ["asia-northeast3", "us-west2", "us-east4"]
} 