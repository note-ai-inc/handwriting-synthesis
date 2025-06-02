output "load_balancer_ip" {
  description = "The IP address of the global load balancer"
  value       = module.global_lb.load_balancer_ip
}

output "http_url" {
  description = "The HTTP URL of the service"
  value       = module.global_lb.http_url
}

output "vm_load_balancer_ip" {
  description = "The IP address of the VM-based global load balancer"
  value       = module.vm_lb.load_balancer_ip
}

output "vm_http_url" {
  description = "The HTTP URL of the VM-based service"
  value       = module.vm_lb.http_url
}

output "vm_https_url" {
  description = "The HTTPS URL of the VM-based service"
  value       = module.vm_lb.https_url
}

output "vm_instances" {
  description = "Map of VM instances by region"
  value       = module.vm.vm_instances
}

output "cloudbuild_service_account_email" {
  description = "The email address of the custom Cloud Build service account."
  value       = module.iam.cloudbuild_service_account_email
} 