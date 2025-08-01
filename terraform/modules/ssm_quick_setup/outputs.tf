output "ssm_association_id" {
  description = "The ID of the SSM Association."
  value       = aws_ssm_association.quick_setup.association_id
}
