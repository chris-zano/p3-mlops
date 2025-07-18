

module "inference_api_repo" {
  source              = "./modules/ecr"
  providers = {
    aws = aws.primary
  }
  repository_name     = var.inference_api_repo_name
}

module "train_script_repo" {
  source              = "./modules/ecr"
  providers = {
    aws = aws.primary
  }
  repository_name     = var.train_script_repo_name
}

module "evaluate_script_repo" {
  source              = "./modules/ecr"
  providers = {
    aws = aws.primary
  }
  repository_name     = var.evaluate_script_repo_name
}


module "project_vpc" {
  source               = "./modules/vpc"
  availability_zones   = var.availability_zones
  vpc_cidr             = var.vpc_cidr
  public_subnet_cidrs  = var.public_subnet_cidrs
  private_subnet_cidrs = var.private_subnet_cidrs

  providers = {
    aws = aws.primary
  }
}

module "mlflow_security_groups" {
  source = "./modules/security_groups"
  providers = {
    aws = aws.primary
  }
  name        = "mlflow_sg"
  vpc_id      = module.project_vpc.vpc_id
  description = "Security group for MLFlow EC2 instance"

  ingress_rules = [
    {
      from_port   = 22
      to_port     = 22
      protocol    = "tcp"
      cidr_blocks = ["0.0.0.0/0"]
    },
    {
      from_port   = 80
      to_port     = 80
      protocol    = "tcp"
      cidr_blocks = ["0.0.0.0/0"]
    }
  ]

  egress_rules = [
    {
      from_port   = 0
      to_port     = 0
      protocol    = "-1"
      cidr_blocks = ["0.0.0.0/0"]
    }
  ]

  tags = {
    Name        = "mlflow-ec2-sg"
    Environment = var.environment
  }
}

module "model_training_security_groups" {
  source = "./modules/security_groups"
  providers = {
    aws = aws.primary
  }
  name        = "model_training_sg"
  vpc_id      = module.project_vpc.vpc_id
  description = "Security group for model_training EC2 instance"

  ingress_rules = [
    {
      from_port   = 22
      to_port     = 22
      protocol    = "tcp"
      cidr_blocks = ["0.0.0.0/0"]
    },
    {
      from_port   = 80
      to_port     = 80
      protocol    = "tcp"
      cidr_blocks = ["0.0.0.0/0"]
    }
  ]

  egress_rules = [
    {
      from_port   = 0
      to_port     = 0
      protocol    = "-1"
      cidr_blocks = ["0.0.0.0/0"]
    }
  ]

  tags = {
    Name        = "mlflow-ec2-sg"
    Environment = var.environment
  }
}

module "inference_api_security_groups" {
  source = "./modules/security_groups"
  providers = {
    aws = aws.primary
  }
  name        = "inference_api_sg"
  vpc_id      = module.project_vpc.vpc_id
  description = "Security group for Inference ECS Service"

  ingress_rules = [
    {
      from_port   = 80
      to_port     = 80
      protocol    = "tcp"
      cidr_blocks = ["0.0.0.0/0"]
    },
    {
      from_port   = 8000
      to_port     = 8000
      protocol    = "tcp"
      cidr_blocks = ["0.0.0.0/0"]
    }
  ]

  egress_rules = [
    {
      from_port   = 0
      to_port     = 0
      protocol    = "-1"
      cidr_blocks = ["0.0.0.0/0"]
    }
  ]

  tags = {
    Name        = "mlflow-api-sg"
    Environment = var.environment
  }
}

module "mlflow_s3_bucket" {
  source = "./modules/s3"
  providers = {
    aws = aws.primary
  }
  source_bucket_name = "mlflow-source-bucket-niico-phase3"
}

module "iam_resources" {
  source = "./modules/iam"
  providers = {
    aws = aws.primary
  }
  mlflow_buckets_arns = [module.mlflow_s3_bucket.s3_bucket_arn]
}

module "datasets" {
  providers = {
    aws = aws.primary
  }
  source = "./modules/data"
  ecr_repo_name = module.train_script_repo.repo_url
}

module "mlflow_instance" {
  source = "./modules/ec2"
  providers = {
    aws = aws.primary
  }
  ami_id               = module.datasets.ubuntu_ami_id
  instance_type        = "t3.medium"
  key_name             = "mlops"
  security_group_ids   = [module.mlflow_security_groups.sg_id]
  subnet_id            = module.project_vpc.public_subnet_ids[0]
  user_data            = module.datasets.mlflow_user_data
  iam_instance_profile = module.iam_resources.mlflow_instance_profile_name
  tags = {
    "Name" = "mlflow-instance"
  }
}

module "model_train_instance" {
  source = "./modules/ec2"
  providers = {
    aws = aws.primary
  }
  ami_id               = module.datasets.ubuntu_ami_id
  # instance_type        = "g4ad.xlarge"
  instance_type        = "t2.micro"
  key_name             = "mlops"
  security_group_ids   = [module.model_training_security_groups.sg_id]
  subnet_id            = module.project_vpc.public_subnet_ids[0]
  user_data            = module.datasets.model_train_user_data
  iam_instance_profile = module.iam_resources.model_training_role
  tags = {
    "Name" = "model-training-instance"
  }
}


module "inference_api_ecs" {
  source = "./modules/ecs"
  providers = {
    aws = aws.primary
  }
  ecs_cluster_name = "inference_api_cluster"
  container_insights_enabled = "enabled"
  
  ecs_td_family = "inference"
  assign_public_ip = true
  container_port = 8000
  cpu_size = 1024
  desired_count = 1
  ecs_service_name = "inference-api"
  ecs_service_sg = [module.inference_api_security_groups.sg_id]
  ecs_service_subnets = module.project_vpc.public_subnet_ids
  host_port = 8000
  image_uri = "084129280516.dkr.ecr.eu-west-1.amazonaws.com/mlops/infer:latest"
  mem_size = 4096
  task_name = "inference-api"
  aws_region = "eu-west-1"
  log_group_name = "inference-api"
  environment_variables = [
    {
      name  = "MFLOW_SERVER_IP"
      value = module.mlflow_instance.public_ip
    },
  ]
}

resource "local_file" "apply_outputs" {
  filename = "outputs.txt"

  content = <<EOT
  "mlflow_instance_public_ip": "${module.mlflow_instance.public_ip}"
  "mlflow_s3_bucket_name": "${module.mlflow_s3_bucket.bucket_name}"
  EOT
}