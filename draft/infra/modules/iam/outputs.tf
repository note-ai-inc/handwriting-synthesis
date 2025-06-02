output "cloudbuild_service_account_email" {
  description = "The email address of the created Cloud Build service account."
  value       = google_service_account.cloudbuild_sa.email
} 