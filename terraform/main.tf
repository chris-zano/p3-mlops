
module "project_vpc" {
 source = "./modules/vpc"
 availability_zones =  var.availability_zones
 vpc_cidr = var.vpc_cidr
 public_subnet_cidrs = var.public_subnet_cidrs
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
  mlflow_buckets_arns = [ module.mlflow_s3_bucket.s3_bucket_arn ]
}

module "datasets" {
   providers = {
   aws = aws.primary
 }
  source = "./modules/data"
}

module "mlflow_instance" {
  source = "./modules/ec2"
   providers = {
   aws = aws.primary
 }
  ami_id = module.datasets.ubuntu_ami_id
  instance_type = "t3.medium"
  key_name = "mlops"
  security_group_ids = [module.mlflow_security_groups.sg_id]
  subnet_id = module.project_vpc.public_subnet_ids[0]
  user_data = module.datasets.mlflow_user_data
  iam_instance_profile  = module.iam_resources.mlflow_instance_profile_name 
  tags = {
    "Name" = "mlflow-instance"
  }
}

resource "local_file" "apply_outputs" {
  filename = "outputs.txt"

  content = <<EOT
  "mlflow_instance_public_ip": "${module.mlflow_instance.public_ip}"
  "mlflow_s3_bucket_name": "${module.mlflow_s3_bucket.bucket_name}"
  EOT
}