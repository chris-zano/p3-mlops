resource "aws_route53_record" "infer" {
  zone_id = var.route53_zone_id
  name    = "infer"
  type    = "CNAME"
  ttl     = 300
  records = [var.target_endpoint]
}


# resource "aws_route53_record" "infer-dev" {
#   zone_id = var.route53_zone_id
#   name    = "infer"
#   type    = "CNAME"
#   ttl     = 5

#   weighted_routing_policy {
#     weight = 10
#   }

#   set_identifier = "dev"
#   records        = ["dev.example.com"] # Replace with actual dev endpoint
# }

# resource "aws_route53_record" "infer-live" {
#   zone_id = var.route53_zone_id
#   name    = "infer"
#   type    = "CNAME"
#   ttl     = 5

#   weighted_routing_policy {
#     weight = 90
#   }

#   set_identifier = "live"
#   records        = ["live.example.com"] # Replace with actual live endpoint
# }
