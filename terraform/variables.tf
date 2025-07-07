variable "vpc_cidr" {
  description = "CIDR block for the VPC (10.0.0.0/16 provides 65,536 IP addresses)"
  type        = string
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for public subnets across multiple AZs"
  type        = list(string)
}

variable "private_subnet_cidrs" {
  description = "CIDR blocks for private subnets across multiple AZs"
  type        = list(string)
}

variable "availability_zones" {
  description = "AWS availability zones in eu-west-1 region"
  type        = list(string)
}

variable "environment" {
  description = "value for the environment tag"
  type        = string
}