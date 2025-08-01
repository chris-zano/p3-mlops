
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


resource "aws_iam_role" "ecr_access" {
  name = "terraformECR"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Effect = "Allow",
      Principal = {
        Service = "ec2.amazonaws.com"
      },
      Action = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role" "model_training_role" {
  name = "model-training-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow",
      Principal = {
        Service = "ec2.amazonaws.com"
      },
      Action = "sts:AssumeRole"
    }]
  })

  tags = {
    Name = "model-training-role"
  }
}

resource "aws_iam_instance_profile" "model_training_profile" {
  name = "model-training-profile"
  role = aws_iam_role.model_training_role.name
}

resource "aws_iam_policy" "ecr_read_only" {
  name = "ecr-read-only"

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ],
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "model_training_ecr_policy_attach" {
  role       = aws_iam_role.model_training_role.name
  policy_arn = aws_iam_policy.ecr_read_only.arn
}

# IAM Role for the Lambda function
resource "aws_iam_role" "lambda_ec2_starter_role" {
  name = var.lambda_role_name

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Action = "sts:AssumeRole",
        Effect = "Allow",
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name = "${var.lambda_role_name}-role"
  }
}

# IAM Policy for Lambda to start EC2 instances and write CloudWatch Logs
resource "aws_iam_policy" "lambda_ec2_starter_policy" {
  name        = "${var.lambda_role_name}-policy"
  description = "IAM policy for Lambda to start EC2 instances and write CloudWatch Logs"

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ],
        Resource = "arn:aws:logs:*:*:*" # Allows logging to any log group in any region
      },
      {
        "Action": [
                "ec2:DescribeInstances",
                "ec2:StartInstances",
                "ec2:StopInstances",
                "ec2:RebootInstances"
            ],
            "Effect": "Allow",
            "Resource": "*"
      },
      {
        Effect = "Allow",
        Action = [
            "ec2:DescribeInstances",
            "ec2:StartInstances",
            "ec2:StopInstances",
            "ec2:RebootInstances"
        ],
        # Restrict to a specific EC2 instance for security
        Resource = "arn:aws:ec2:${var.aws_region}:${var.aws_account_id}:instance/${var.ec2_instance_id}"
      }
    ]
  })
}

# Attach the policy to the role
resource "aws_iam_role_policy_attachment" "lambda_ec2_starter_attachment" {
  role       = aws_iam_role.lambda_ec2_starter_role.name
  policy_arn = aws_iam_policy.lambda_ec2_starter_policy.arn
}

# Add this new role and policy to your existing modules/iam/main.tf

# IAM Role for the challenger redeployment Lambda function
resource "aws_iam_role" "challenger_redeployment_role" {
  name = "${var.project_name}-challenger-redeployment-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Action = "sts:AssumeRole",
        Effect = "Allow",
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })
}

# IAM Policy to allow the Lambda to update ECS services and write logs
resource "aws_iam_policy" "challenger_redeployment_policy" {
  name        = "${var.project_name}-challenger-redeployment-policy"
  description = "IAM policy for Lambda to update ECS service and write logs"

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ],
        Resource = "arn:aws:logs:*:*:*"
      },
      {
        Effect = "Allow",
        Action = [
          "ecs:UpdateService",
          "ecs:DescribeServices",
          "ecs:DescribeTaskDefinition"
        ],
        Resource = [
          "arn:aws:ecs:${var.aws_region}:${var.aws_account_id}:service/${var.ecs_cluster_name}/${var.inference_challenger_ecs_service_name}",
          "arn:aws:ecs:${var.aws_region}:${var.aws_account_id}:cluster/${var.ecs_cluster_name}"
        ]
      }
    ]
  })
}

# Attach the policy to the role
resource "aws_iam_role_policy_attachment" "challenger_redeployment_attachment" {
  role       = aws_iam_role.challenger_redeployment_role.name
  policy_arn = aws_iam_policy.challenger_redeployment_policy.arn
}
