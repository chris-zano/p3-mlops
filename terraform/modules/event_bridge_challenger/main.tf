terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
    }
  }
}

# Defines an EventBridge rule that triggers a Lambda function on an ECR image push with a specific tag.
resource "aws_cloudwatch_event_rule" "challenger_repo_push_rule" {
  name        = var.rule_name
  description = "Triggers on successful ECR image push to the challenger repository with a specific tag."

  # The event pattern now includes a filter for the 'image-tag'
  event_pattern = jsonencode({
    "source": ["aws.ecr"],
    "detail-type": ["ECR Image Action"],
    "detail": {
      "action-type": ["PUSH"],
      "result": ["SUCCESS"],
      "repository-name": [var.ecr_repository_name],
      "image-tag": [var.ecr_image_tag]
    }
  })

  tags = {
    Name = "${var.rule_name}-rule"
  }
}

# Defines the target for the EventBridge rule, which is the redeployment Lambda function.
resource "aws_cloudwatch_event_target" "challenger_redeploys_lambda_target" {
  rule      = aws_cloudwatch_event_rule.challenger_repo_push_rule.name
  target_id = "redeploys-challenger-lambda"
  arn       = var.lambda_function_arn
}

