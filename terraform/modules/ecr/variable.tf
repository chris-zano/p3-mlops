variable "repository_name" {
  type = string
  description = "Name of the ecr repository"
}

variable "images_to_keep" {
  type = number
  description = "Number of container images, to retain"
}