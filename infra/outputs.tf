output "s3_bucket_name" {
  value = aws_s3_bucket.model_storage.id
}

output "dynamodb_table_name" {
  value = aws_dynamodb_table.model_metadata.name
}

output "cloudfront_domain_name" {
  value = aws_cloudfront_distribution.model_cdn.domain_name
}

output "sqs_queue_url" {
  value = aws_sqs_queue.training_complete.url
  description = "URL of the SQS queue for training notifications"
}

output "sqs_queue_arn" {
  value = aws_sqs_queue.training_complete.arn
  description = "ARN of the SQS queue for training notifications"
} 