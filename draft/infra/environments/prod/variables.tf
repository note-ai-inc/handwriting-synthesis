variable "project_id" {
  description = "The GCP project ID"
  type        = string
}

variable "region" {
  description = "The default region"
  type        = string
}

variable "regions" {
  description = "List of regions to deploy to"
  type        = list(string)
}

variable "domain_name" {
  description = "The domain name for the service"
  type        = string
}

variable "vm_domain_name" {
  description = "The domain name for the VM service"
  type        = string
}

variable "service_account_id" {
  description = "The service account ID (project number)"
  type        = string
}

variable "repository_url" {
  description = "The Git repository URL"
  type        = string
} 