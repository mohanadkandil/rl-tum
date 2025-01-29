resource "aws_s3_bucket" "model_storage" {
  bucket        = "${var.project_name}-models-${var.environment}-${formatdate("YYYYMMDD", timestamp())}"
  force_destroy = true

  timeouts {
    create = "5m"
    delete = "5m"
  }
}

resource "random_string" "bucket_suffix" {
  length  = 8
  special = false
  upper   = false
}

resource "aws_dynamodb_table" "model_metadata" {
  name           = "${var.project_name}-model-metadata-${var.environment}"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "model_id"

  attribute {
    name = "model_id"
    type = "S"
  }

  tags = {
    Name        = "Model Metadata"
    Environment = var.environment
  }
}

resource "aws_cloudfront_distribution" "model_cdn" {
  origin {
    domain_name = aws_s3_bucket.model_storage.bucket_regional_domain_name
    origin_id   = "S3Origin"
  }
  enabled = true
  default_cache_behavior {
    target_origin_id = "S3Origin"
    allowed_methods  = ["GET", "HEAD"]
    cached_methods   = ["GET", "HEAD"]
    forwarded_values {
      query_string = false
      cookies {
        forward = "none"
      }
    }
    viewer_protocol_policy = "redirect-to-https"
  }
  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }
  viewer_certificate {
    cloudfront_default_certificate = true
  }
}

resource "aws_s3_bucket_policy" "allow_cloudfront" {
  bucket = aws_s3_bucket.model_storage.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "AllowCloudFrontAccess"
        Effect    = "Allow"
        Principal = {
          Service = "cloudfront.amazonaws.com"
        }
        Action   = "s3:GetObject"
        Resource = "${aws_s3_bucket.model_storage.arn}/*"
        Condition = {
          StringEquals = {
            "AWS:SourceArn" = aws_cloudfront_distribution.model_cdn.arn
          }
        }
      }
    ]
  })
}

resource "aws_sqs_queue" "training_complete" {
  name = "${var.project_name}-training-complete-${var.environment}"
  message_retention_seconds = 86400  # 1 day
}

resource "aws_sqs_queue_policy" "training_complete_policy" {
  queue_url = aws_sqs_queue.training_complete.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "s3.amazonaws.com"
        }
        Action = "sqs:SendMessage"
        Resource = aws_sqs_queue.training_complete.arn
        Condition = {
          ArnLike = {
            "aws:SourceArn": aws_s3_bucket.model_storage.arn
          }
        }
      }
    ]
  })
}

# API Gateway for model uploads
resource "aws_api_gateway_rest_api" "model_api" {
  name = "checkers-ai-api"
}

# API Resource for models
resource "aws_api_gateway_resource" "models" {
  rest_api_id = aws_api_gateway_rest_api.model_api.id
  parent_id   = aws_api_gateway_rest_api.model_api.root_resource_id
  path_part   = "models"
}

# POST method for uploading models
resource "aws_api_gateway_method" "upload" {
  rest_api_id   = aws_api_gateway_rest_api.model_api.id
  resource_id   = aws_api_gateway_resource.models.id
  http_method   = "POST"
  authorization = "NONE"
  api_key_required = true
}

# Create API key
resource "aws_api_gateway_api_key" "upload_key" {
  name = "checkers-ai-upload-key"
}

# Create usage plan
resource "aws_api_gateway_usage_plan" "upload_plan" {
  name = "checkers-ai-upload-plan"

  api_stages {
    api_id = aws_api_gateway_rest_api.model_api.id
    stage  = aws_api_gateway_stage.api.stage_name
  }
}

resource "aws_api_gateway_usage_plan_key" "upload_key" {
  key_id        = aws_api_gateway_api_key.upload_key.id
  key_type      = "API_KEY"
  usage_plan_id = aws_api_gateway_usage_plan.upload_plan.id
}

resource "aws_lambda_function" "upload_handler" {
  filename      = "upload_handler.zip"
  function_name = "model-upload-handler"
  role         = aws_iam_role.lambda_role.arn
  handler      = "index.handler"
  runtime      = "python3.9"
}

resource "aws_iam_role" "lambda_role" {
  name = "checkers-ai-upload-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "lambda_policy" {
  name = "checkers-ai-upload-lambda-policy"
  role = aws_iam_role.lambda_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:PutObject",
          "dynamodb:PutItem",
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = [
          "${aws_s3_bucket.model_storage.arn}/*",
          aws_dynamodb_table.model_metadata.arn,
          "arn:aws:logs:*:*:*"
        ]
      }
    ]
  })
}

resource "aws_api_gateway_deployment" "api" {
  rest_api_id = aws_api_gateway_rest_api.model_api.id
  
  depends_on = [
    aws_api_gateway_method.upload,
    aws_api_gateway_integration.lambda
  ]

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_api_gateway_stage" "api" {
  deployment_id = aws_api_gateway_deployment.api.id
  rest_api_id   = aws_api_gateway_rest_api.model_api.id
  stage_name    = "dev"
}

# Add this after the upload method
resource "aws_api_gateway_integration" "lambda" {
  rest_api_id = aws_api_gateway_rest_api.model_api.id
  resource_id = aws_api_gateway_resource.models.id
  http_method = aws_api_gateway_method.upload.http_method
  
  integration_http_method = "POST"
  type                   = "AWS_PROXY"
  uri                    = aws_lambda_function.upload_handler.invoke_arn
}

# Add Lambda permission to allow API Gateway
resource "aws_lambda_permission" "apigw" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.upload_handler.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_api_gateway_rest_api.model_api.execution_arn}/*/*"
}

output "api_key_id" {
  value = aws_api_gateway_api_key.upload_key.id
}

output "api_endpoint" {
  value = "${aws_api_gateway_deployment.api.invoke_url}${aws_api_gateway_stage.api.stage_name}/models"
}
