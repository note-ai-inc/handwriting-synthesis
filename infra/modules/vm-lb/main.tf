terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

# Create static IP for the global load balancer
resource "google_compute_global_address" "default" {
  provider = google
  project  = var.project_id
  name     = "${var.service_name}-vm-ip"
  description = "Static IP for the VM-based global load balancer"
}

# Create health check
resource "google_compute_health_check" "default" {
  provider = google
  project  = var.project_id
  name     = "${var.service_name}-health-check"
  
  http_health_check {
    port         = 80
    request_path = "/health"
  }
}

# Create instance groups for each VM
resource "google_compute_instance_group" "default" {
  for_each = { for vm_name, vm_self_link in var.vm_self_links : vm_name => vm_self_link }
  
  name      = "${var.service_name}-ig-${each.key}"
  project   = var.project_id
  zone      = "${each.key}-a"
  
  instances = [each.value]
  
  # Add named port for HTTP traffic
  named_port {
    name = "http"
    port = 80
  }
}

# Create backend service with geo routing
resource "google_compute_backend_service" "default" {
  provider = google
  project  = var.project_id
  name     = "${var.service_name}-backend"
  health_checks = [google_compute_health_check.default.self_link]
  port_name = "http"
  protocol = "HTTP"
  
  dynamic "backend" {
    for_each = google_compute_instance_group.default
    content {
      group = backend.value.self_link
      capacity_scaler = 1.0
      balancing_mode = "UTILIZATION"
      max_utilization = 0.8
    }
  }
  
  # Enable connection draining
  connection_draining_timeout_sec = 300
  
  # Add custom headers for original client IP and geo information
  custom_request_headers = [
    "X-Client-Geo-Location: {client_region_subdivision}",
    "X-Client-City: {client_city}",
    "X-Client-Region: {client_region}"
  ]
  
  log_config {
    enable = true
    sample_rate = 1.0
  }
  
  # Set timeouts
  timeouts {
    create = "5m"
    update = "5m"
  }
}

# Create URL map for HTTPS traffic
resource "google_compute_url_map" "default" {
  provider = google
  project  = var.project_id
  name     = "${var.service_name}-urlmap"
  
  # Default to the redirect when no host rules match (handles IP access)
  default_url_redirect {
    host_redirect = var.domain_name
    https_redirect = true
    redirect_response_code = "MOVED_PERMANENTLY_DEFAULT"
    strip_query = false
  }
  
  # Only allow access to the backend service when the host matches the domain name
  host_rule {
    hosts = [var.domain_name]
    path_matcher = "domain-matcher"
  }
  
  path_matcher {
    name = "domain-matcher"
    default_service = google_compute_backend_service.default.id
  }
}

# Create URL map for HTTP to HTTPS redirect
resource "google_compute_url_map" "http_redirect" {
  provider = google
  project  = var.project_id
  name     = "${var.service_name}-http-redirect"
  
  default_url_redirect {
    host_redirect = var.domain_name
    https_redirect = true
    redirect_response_code = "MOVED_PERMANENTLY_DEFAULT"
    strip_query = false
  }
}

# HTTP proxy for redirect
resource "google_compute_target_http_proxy" "default" {
  provider = google
  project  = var.project_id
  name     = "${var.service_name}-http-proxy"
  url_map  = google_compute_url_map.http_redirect.id
}

# HTTP forwarding rule
resource "google_compute_global_forwarding_rule" "default" {
  provider = google
  project  = var.project_id
  name     = "${var.service_name}-lb"
  target   = google_compute_target_http_proxy.default.id
  port_range = "80"
  ip_protocol = "TCP"
  ip_address = google_compute_global_address.default.address
}

# Create SSL certificate
resource "google_compute_managed_ssl_certificate" "default" {
  provider = google
  project  = var.project_id
  name     = "${var.service_name}-cert"
  
  managed {
    domains = [var.domain_name]
  }
}

# Create HTTPS proxy
resource "google_compute_target_https_proxy" "https" {
  provider = google
  project  = var.project_id
  name     = "${var.service_name}-https-proxy"
  url_map  = google_compute_url_map.default.id
  ssl_certificates = [google_compute_managed_ssl_certificate.default.id]
}

# Create HTTPS forwarding rule
resource "google_compute_global_forwarding_rule" "https" {
  provider = google
  project  = var.project_id
  name     = "${var.service_name}-https-lb"
  target   = google_compute_target_https_proxy.https.id
  port_range = "443"
  ip_protocol = "TCP"
  ip_address = google_compute_global_address.default.address
} 