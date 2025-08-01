output "event_bridge_rule_arn" {
  description = "The ARN of the EventBridge rule."
  value       = aws_cloudwatch_event_rule.challenger_repo_push_rule.arn
}
