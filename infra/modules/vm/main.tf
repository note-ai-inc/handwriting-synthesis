terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

# Create a dedicated service account for the VMs
resource "google_service_account" "vm_service_account" {
  account_id   = "${var.service_name}-vm-sa"
  display_name = "${var.service_name} VM Service Account"
  project      = var.project_id
}

# Grant Container Registry (GCR) reader role (access to underlying GCS buckets)
# Note: This is for GCR, not Artifact Registry.
resource "google_project_iam_member" "gcr_reader" {
  project = var.project_id
  role    = "roles/storage.objectViewer"
  member  = "serviceAccount:${google_service_account.vm_service_account.email}"
}

# Grant Artifact Registry reader role
resource "google_project_iam_member" "artifact_registry_reader" {
  project = var.project_id
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:${google_service_account.vm_service_account.email}"
}

# Create VMs in each region
resource "google_compute_instance" "vm" {
  for_each = toset(var.regions)
  
  name         = "${var.service_name}-${each.value}"
  machine_type = "c2d-highcpu-32"
  zone         = "${each.value}-a"
  project      = var.project_id
  
  # Ensure IAM roles and Firewalls are created before instance creation
  depends_on = [
    google_project_iam_member.gcr_reader,
    google_project_iam_member.artifact_registry_reader,
    google_compute_firewall.allow_http_https,
    google_compute_firewall.allow_health_check
  ]

  boot_disk {
    auto_delete = true
    device_name = "${var.service_name}-${each.value}"

    initialize_params {
      image = "projects/debian-cloud/global/images/family/debian-12"
      size  = 100
      type  = "pd-ssd"
    }

    mode = "READ_WRITE"
  }

  network_interface {
    subnetwork = "projects/${var.project_id}/regions/${each.value}/subnetworks/default"
    
    access_config {
      network_tier = "PREMIUM"
    }
  }

  service_account {
    # Use the dedicated service account created above
    email  = google_service_account.vm_service_account.email
    scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
  }

  scheduling {
    automatic_restart   = true
    on_host_maintenance = "MIGRATE"
    preemptible         = false
    provisioning_model  = "STANDARD"
  }

  shielded_instance_config {
    enable_integrity_monitoring = true
    enable_secure_boot          = false
    enable_vtpm                 = true
  }

  tags = ["http-server", "https-server", "lb-health-check"]

  # Add labels for ops agent
  labels = {
    goog-ec-src = "vm_add-tf"
    # Remove Ops agent label
    # goog-ops-agent-policy = "v2-x86-template-1-4-0"
  }

  metadata = {
    enable-osconfig = "TRUE"
    ssh-keys = "deploy:${var.ssh_public_key}"  # Keep SSH key
    # Replace startup script with Docker installation
    startup-script = <<-EOF
      #!/bin/bash
      set -e # Exit immediately if a command exits with a non-zero status.
      exec > >(tee /var/log/startup-script.log|logger -t startup-script -s 2>/dev/console) 2>&1

      echo "[$(date)] Starting Docker installation script..."

      # 1. Set up the repository
      echo "[$(date)] Updating apt package index..."
      apt-get update -y
      echo "[$(date)] Installing prerequisite packages..."
      apt-get install -y ca-certificates curl gnupg

      echo "[$(date)] Creating directory for apt keyrings..."
      install -m 0755 -d /etc/apt/keyrings
      echo "[$(date)] Downloading Docker's official GPG key..."
      curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
      chmod a+r /etc/apt/keyrings/docker.gpg

      echo "[$(date)] Setting up the Docker repository..."
      echo \
        "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian \
        $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
        tee /etc/apt/sources.list.d/docker.list > /dev/null

      echo "[$(date)] Updating apt package index again after adding Docker repo..."
      apt-get update -y

      # 2. Install Docker Engine
      echo "[$(date)] Installing Docker Engine, CLI, containerd, buildx, and compose plugin..."
      apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

      # 3. Verify installation (optional, but good practice)
      echo "[$(date)] Verifying Docker installation by running hello-world..."
      if docker run hello-world; then
        echo "[$(date)] Docker hello-world container ran successfully."
      else
        echo "[$(date)] ERROR: Docker hello-world container failed to run."
        # Consider adding error handling/exit here if critical
      fi

      # 4. Post-installation steps: Ensure Docker starts on boot
      echo "[$(date)] Enabling Docker service to start on boot..."
      systemctl enable docker
      # Ensure the service is started now as well
      systemctl start docker
      echo "[$(date)] Docker service enabled and started."

      # Note: To run docker commands without sudo, add user to docker group:
      # usermod -aG docker your-user
      # This script runs as root, so sudo isn't needed for docker commands here.

      echo "[$(date)] Docker installation script completed."
    EOF
  }
}

# Create a firewall rule for HTTP/HTTPS traffic
resource "google_compute_firewall" "allow_http_https" {
  name    = "${var.service_name}-allow-http-https"
  project = var.project_id
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["80", "443", "8000"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["http-server", "https-server"]
}

# Create a firewall rule for health checks
resource "google_compute_firewall" "allow_health_check" {
  name    = "${var.service_name}-allow-health-check"
  project = var.project_id
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["80", "443"]
  }

  source_ranges = ["35.191.0.0/16", "130.211.0.0/22"]
  target_tags   = ["lb-health-check"]
}