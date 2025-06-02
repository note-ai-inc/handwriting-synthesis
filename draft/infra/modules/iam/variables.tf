variable "project_id" {
  description = "The GCP project ID where resources will be created."
  type        = string
}

variable "cloudbuild_service_account_id" {
  description = "The desired account ID for the Cloud Build service account (e.g., 'my-cloudbuild-sa')."
  type        = string
  default     = "cloudbuild-custom-sa"
}

variable "cloudbuild_service_account_display_name" {
  description = "The display name for the Cloud Build service account."
  type        = string
  default     = "Custom Cloud Build Service Account"
} 