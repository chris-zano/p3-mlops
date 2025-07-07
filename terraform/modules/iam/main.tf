
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
    }
  }
}

resource "aws_iam_role" "mlflow_instance_role" {
  name = "mlflow-instance-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = {
    Name = "mlflow-instance-role"
  }
}

resource "aws_iam_instance_profile" "mlflow_instance_profile" {
  name = "mlflow-instance-profile"
  role = aws_iam_role.mlflow_instance_role.name
}

resource "aws_iam_policy" "mlflow_s3_bucket_access" {
  name = "mlflow-s3-bucket-access"

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Action = [
          "s3:PutObject",
          "s3:GetObject",
          "s3:ListBucket"
        ],
        Resource = var.mlflow_buckets_arns
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "mlflow_s3_bucket_policy_attach" {
  role       = aws_iam_role.mlflow_instance_role.name
  policy_arn = aws_iam_policy.mlflow_s3_bucket_access.arn
}
