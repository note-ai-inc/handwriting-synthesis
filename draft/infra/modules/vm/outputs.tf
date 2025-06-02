output "vm_instances" {
  description = "Map of VM instances by region"
  value = {
    for region, instance in google_compute_instance.vm : region => {
      name = instance.name
      id   = instance.id
      self_link = instance.self_link
      network_interface = instance.network_interface[0].network_ip
      external_ip = instance.network_interface[0].access_config[0].nat_ip
    }
  }
}

output "vm_names" {
  description = "Map of VM names by region"
  value = {
    for region, instance in google_compute_instance.vm : region => instance.name
  }
}

output "vm_self_links" {
  description = "Map of VM self links by region"
  value = {
    for region, instance in google_compute_instance.vm : region => instance.self_link
  }
} 