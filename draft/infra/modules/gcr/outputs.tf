output "repository_urls" {
  description = "Map of region to repository URLs"
  value = {
    for region, repo in google_artifact_registry_repository.repositories : region => repo.name
  }
}

output "repository_ids" {
  description = "Map of region to repository IDs"
  value = {
    for region, repo in google_artifact_registry_repository.repositories : region => repo.repository_id
  }
} 