terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

# 1. Create Serverless NEGs for each region
resource "google_compute_region_network_endpoint_group" "serverless_neg" {
  for_each = toset(var.regions)
  
  provider               = google
  project               = var.project_id
  name                  = "neg-${var.service_name}-${each.value}"
  network_endpoint_type = "SERVERLESS"
  region                = each.value
  
  cloud_run {
    service = "${var.service_name}-${each.value}"
  }
}

# 2. Create backend service with geo routing
resource "google_compute_backend_service" "default" {
  provider = google
  project  = var.project_id
  name     = "${var.service_name}-backend"

  dynamic "backend" {
    for_each = google_compute_region_network_endpoint_group.serverless_neg
    content {
      group = backend.value.id
    }
  }

  # Enable Cloud CDN for better performance
  enable_cdn = true
  
  # Add custom headers for original client IP and geo information
  custom_request_headers = [
    "X-Client-Geo-Location: {client_region_subdivision}",
    "X-Client-City: {client_city}",
    "X-Client-Region: {client_region}"
  ]
}

# Static IP (existing)
resource "google_compute_global_address" "default" {
  provider = google
  project  = var.project_id
  name     = "${var.service_name}-ip"
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

# HTTP forwarding rule (updated to only use static IP)
resource "google_compute_global_forwarding_rule" "default" {
  provider = google
  project  = var.project_id
  name     = "${var.service_name}-lb"  # Keep the same name to ensure proper replacement
  target   = google_compute_target_http_proxy.default.id
  port_range = "80"
  ip_protocol = "TCP"
  ip_address = google_compute_global_address.default.address  # Use only the static IP
}

# Create SSL certificate
resource "google_compute_managed_ssl_certificate" "default" {
  provider = google
  project  = var.project_id
  name     = "${var.service_name}-cert-new"  # New name to avoid conflicts
  
  managed {
    domains = [var.domain_name]
  }
}

# Create HTTPS proxy
resource "google_compute_target_https_proxy" "https" {
  name             = "${var.service_name}-https-proxy"
  url_map          = google_compute_url_map.default.id
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