terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

# Create the custom service account for Cloud Build
resource "google_service_account" "cloudbuild_sa" {
  account_id   = var.cloudbuild_service_account_id
  display_name = var.cloudbuild_service_account_display_name
  project      = var.project_id
}

# --- Grant necessary roles to the Cloud Build service account ---

# Role for managing Cloud Build resources
resource "google_project_iam_member" "cloudbuild_editor" {
  project = var.project_id
  role    = "roles/cloudbuild.builds.editor"
  member  = "serviceAccount:${google_service_account.cloudbuild_sa.email}"
}

# Role for pushing to Artifact Registry (needs storage admin on GCS buckets)
# Alternatively, use roles/artifactregistry.writer if you only use Artifact Registry directly
resource "google_project_iam_member" "artifact_registry_writer" {
  project = var.project_id
  role    = "roles/storage.objectAdmin" # Broad permission for GCR/Artifact Registry
  member  = "serviceAccount:${google_service_account.cloudbuild_sa.email}"
}

# Role for deploying to Cloud Run
resource "google_project_iam_member" "run_admin" {
  project = var.project_id
  role    = "roles/run.admin"
  member  = "serviceAccount:${google_service_account.cloudbuild_sa.email}"
}

# Role for SSH access via OS Login
resource "google_project_iam_member" "compute_os_login" {
  project = var.project_id
  role    = "roles/compute.osLogin"
  member  = "serviceAccount:${google_service_account.cloudbuild_sa.email}"
}

# Role to allow Cloud Build SA to act as other Service Accounts 
# (e.g., the Cloud Run service account during deployment)
resource "google_project_iam_member" "iam_service_account_user" {
  project = var.project_id
  role    = "roles/iam.serviceAccountUser"
  member  = "serviceAccount:${google_service_account.cloudbuild_sa.email}"
} 