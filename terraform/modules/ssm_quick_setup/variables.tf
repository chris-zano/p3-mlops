variable "ssm_document_name" {
  description = "The name of the AWS Systems Manager document to use (e.g., 'AWS-StartEC2Instance' or 'AWS-StopEC2Instance')."
  type        = string
}

variable "schedule_expression" {
  description = "The cron or rate expression for the SSM association schedule."
  type        = string
  # Examples:
  # "cron(0 9 ? * MON-FRI *)" -> At 9 AM on weekdays
  # "cron(0 18 ? * MON-FRI *)" -> At 6 PM on weekdays
  # "rate(1 hour)" -> Every hour
}

variable "instance_tag_key" {
  description = "The tag key used to identify the target EC2 instance(s)."
  type        = string
  default     = "QuickSetup"
}

variable "instance_tag_value" {
  description = "The tag value used to identify the target EC2 instance(s)."
  type        = string
}
