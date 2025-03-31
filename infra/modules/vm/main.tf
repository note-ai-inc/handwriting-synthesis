terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

# Create VMs in each region
resource "google_compute_instance" "vm" {
  for_each = toset(var.regions)
  
  name         = "${var.service_name}-${each.value}"
  machine_type = "c2d-standard-16"
  zone         = "${each.value}-a"  # Use 'a' zone in each region
  project      = var.project_id
  
  boot_disk {
    auto_delete = true
    device_name = "${var.service_name}-${each.value}"

    initialize_params {
      image = "projects/debian-cloud/global/images/debian-12-bookworm-v20250311"
      size  = 100
      type  = "pd-ssd"
    }

    mode = "READ_WRITE"
  }

  network_interface {
    subnetwork = "projects/${var.project_id}/regions/${each.value}/subnetworks/default"
    
    access_config {
      network_tier = "PREMIUM"
      # This creates an ephemeral external IP
    }
  }

  service_account {
    email  = "${var.service_account_id}-compute@developer.gserviceaccount.com"
    scopes = [
      "https://www.googleapis.com/auth/devstorage.read_only", 
      "https://www.googleapis.com/auth/logging.write", 
      "https://www.googleapis.com/auth/monitoring.write", 
      "https://www.googleapis.com/auth/service.management.readonly", 
      "https://www.googleapis.com/auth/servicecontrol", 
      "https://www.googleapis.com/auth/trace.append"
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
    goog-ops-agent-policy = "v2-x86-template-1-4-0"
  }

  metadata = {
    enable-osconfig = "TRUE"
    ssh-keys = "deploy:${var.ssh_public_key}"  # Add SSH key for deploy user
    startup-script = <<-EOF
      #!/bin/bash
      set -e

      echo "[$(date)] Starting startup script..."

      # Set HOME environment variable
      export HOME=/root

      echo "[$(date)] Checking if Docker is already installed..."
      if command -v docker &> /dev/null; then
        echo "[$(date)] Docker is already installed. Version: $(docker --version)"
        echo "[$(date)] Skipping Docker installation..."
      else
        echo "[$(date)] Docker is not installed. Proceeding with installation..."
        echo "[$(date)] Installing required packages..."
        apt-get update
        apt-get install -y \
          apt-transport-https \
          ca-certificates \
          curl \
          gnupg \
          lsb-release \
          git \
          google-cloud-cli

        echo "[$(date)] Installing Docker..."
        # Add Docker repository and key in a non-interactive way
        echo "[$(date)] Setting up Docker repository..."
        install -m 0755 -d /etc/apt/keyrings
        echo "[$(date)] Downloading Docker GPG key..."
        curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
        chmod a+r /etc/apt/keyrings/docker.gpg

        echo "[$(date)] Adding Docker repository to apt sources..."
        # Add the repository to Apt sources
        echo \
          "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian \
          "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
          tee /etc/apt/sources.list.d/docker.list > /dev/null

        echo "[$(date)] Installing Docker packages..."
        apt-get update
        apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
      fi

      echo "[$(date)] Configuring Docker..."
      # Create docker group and add user
      groupadd -f docker
      usermod -aG docker root
      usermod -aG docker deploy

      echo "[$(date)] Setting up application..."
      # Create application directory if it doesn't exist
      mkdir -p /home/deploy/app
      chown -R deploy:deploy /home/deploy/app

      echo "[$(date)] Configuring Git..."
      # Configure Git for deploy user
      su - deploy -c 'git config --global --add safe.directory /home/deploy/app/handwriting-synthesis'
      su - deploy -c 'git config --global credential.helper store'

      echo "[$(date)] Pulling latest changes..."
      # Pull latest changes as deploy user
      su - deploy -c 'cd /home/deploy/app/handwriting-synthesis && git pull'

      echo "[$(date)] Building and running Docker container..."
      # Build and run Docker container
      cd /home/deploy/app/handwriting-synthesis
      docker build -t handwriting-synthesis .
      
      # Stop and remove any existing containers
      echo "[$(date)] Checking for existing containers..."
      if [ "$(docker ps -q --filter ancestor=handwriting-synthesis)" ]; then
        echo "[$(date)] Stopping existing containers..."
        docker stop $(docker ps -q --filter ancestor=handwriting-synthesis)
      fi
      
      if [ "$(docker ps -aq --filter ancestor=handwriting-synthesis)" ]; then
        echo "[$(date)] Removing stopped containers..."
        docker rm $(docker ps -aq --filter ancestor=handwriting-synthesis)
      fi
      
      echo "[$(date)] Starting new container..."
      docker run -d -p 80:8000 --restart always handwriting-synthesis

      echo "[$(date)] Startup script completed successfully"
    EOF
  }

  depends_on = [google_compute_firewall.allow_http_https]
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