variable "project_id" {
  description = "The GCP project ID"
  type        = string
}

variable "regions" {
  description = "List of regions to create repositories in"
  type        = list(string)
  default     = ["asia-northeast3", "us-west2", "us-east4"]
} 