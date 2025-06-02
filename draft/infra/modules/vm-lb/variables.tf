variable "project_id" {
  description = "The GCP project ID"
  type        = string
}

variable "service_name" {
  description = "Name of the service (used for resource naming)"
  type        = string
  default     = "handwriting-synthesis"
}

variable "regions" {
  description = "List of regions to deploy VMs in"
  type        = list(string)
  default     = ["asia-northeast3", "us-west2", "us-east4"]
}

variable "domain_name" {
  description = "Domain name for the service (e.g., vm.synthesis.tricklau.xyz)"
  type        = string
}

variable "vm_self_links" {
  description = "Map of VM self links by region"
  type        = map(string)
} 