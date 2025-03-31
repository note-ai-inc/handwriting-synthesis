output "load_balancer_ip" {
  description = "The static IP address of the VM global load balancer"
  value       = google_compute_global_address.default.address
}

output "http_url" {
  description = "The HTTP URL of the service"
  value       = "http://${var.domain_name}"
}

output "https_url" {
  description = "The HTTPS URL of the service"
  value       = "https://${var.domain_name}"
} 