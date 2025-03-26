variable "project_id" {
  description = "The GCP project ID"
  type        = string
}

variable "region" {
  description = "The default GCP region"
  type        = string
  default     = "asia-northeast3"
}

variable "regions" {
  description = "List of regions to create repositories in"
  type        = list(string)
  default     = ["asia-northeast3", "us-west2", "us-east4"]
}

variable "domain_name" {
  description = "Domain name for the service"
  type        = string
  default     = "synthesis.tricklau.xyz"
} 