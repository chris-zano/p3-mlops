terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
    }
  }
}

module "ecr" {
  source = "terraform-aws-modules/ecr/aws"

  repository_name = var.repository_name

  repository_read_write_access_arns = ["${var.ecr_access_role_arn}"]
  repository_lifecycle_policy = jsonencode({
    rules = [
      {
        rulePriority = 1,
        description  = "Keep last 3 images",
        selection = {
          tagStatus     = "tagged",
          tagPrefixList = ["v"],
          countType     = "imageCountMoreThan",
          countNumber   = var.images_to_keep
        },
        action = {
          type = "expire"
        }
      }
    ]
  })

  tags = {
    Terraform   = "true"
    Environment = "dev"
  }
}