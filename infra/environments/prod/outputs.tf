output "load_balancer_ip" {
  description = "The IP address of the global load balancer"
  value       = module.global_lb.load_balancer_ip
}

output "http_url" {
  description = "The HTTP URL of the service"
  value       = module.global_lb.http_url
} 