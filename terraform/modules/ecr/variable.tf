variable "repository_name" {
  type = string
  description = "Name of the ecr repository"
}

variable "images_to_keep" {
  type = number
  description = "Number of container images, to retain"
}

variable "account_id" {
  type = number
  description = "account id for the aws account where the resources are being provisioned"
}

variable "ecr_access_role_arn" {
  type = string
  description = "the arn of the ecr access role"
}