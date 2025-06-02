variable "project_id" {
  description = "The GCP project ID"
  type        = string
}

variable "regions" {
  description = "List of regions to create repositories in"
  type        = list(string)
}

variable "service_name" {
  description = "Name of the service (used for resource naming)"
  type        = string
}

variable "domain_name" {
  description = "Domain name for the service (e.g., synthesis.tricklau.xyz)"
  type        = string
} 