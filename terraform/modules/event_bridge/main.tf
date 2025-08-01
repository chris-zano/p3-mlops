terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
    }
  }
}

resource "aws_cloudwatch_event_rule" "ecr_push_rule" {
  name        = var.rule_name
  description = "Triggers on successful ECR image push to a specific repository."

  event_pattern = jsonencode({
    "source": ["aws.ecr"],
    "detail-type": ["ECR Image Action"],
    "detail": {
      "action-type": ["PUSH"],
      "result": ["SUCCESS"],
      "repository-name": [var.ecr_repository_name]
      # Optional: "image-tag": ["latest"] if you want to be more specific
    }
  })

  tags = {
    Name = "${var.rule_name}-rule"
  }
}



# EventBridge Target: Lambda Function
resource "aws_cloudwatch_event_target" "lambda_target" {
  rule      = aws_cloudwatch_event_rule.ecr_push_rule.name
  target_id = "start-ec2-lambda"
  arn       = var.lambda_function_arn
}
