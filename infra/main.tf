resource "aws_s3_bucket" "model_storage" {
  bucket        = "checkers-ai-models-dev-20250129"
  force_destroy = true
}

# Disable block public access
resource "aws_s3_bucket_public_access_block" "model_storage" {
  bucket = aws_s3_bucket.model_storage.id

  block_public_acls       = false
  block_public_policy     = false
  ignore_public_acls      = false
  restrict_public_buckets = false
}

# Add CORS configuration
resource "aws_s3_bucket_cors_configuration" "model_storage" {
  bucket = aws_s3_bucket.model_storage.id

  cors_rule {
    allowed_headers = ["*"]
    allowed_methods = ["PUT", "POST", "GET"]
    allowed_origins = ["*"]
    max_age_seconds = 3000
  }

  depends_on = [aws_s3_bucket_public_access_block.model_storage]
}

# Add bucket policy to allow uploads
resource "aws_s3_bucket_policy" "allow_uploads" {
  bucket = aws_s3_bucket.model_storage.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "AllowUpload"
        Effect    = "Allow"
        Principal = "*"
        Action    = ["s3:PutObject", "s3:GetObject"]
        Resource  = ["${aws_s3_bucket.model_storage.arn}/*"]
      }
    ]
  })

  depends_on = [aws_s3_bucket_public_access_block.model_storage]
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

# First create deployment
resource "aws_api_gateway_deployment" "api" {
  rest_api_id = aws_api_gateway_rest_api.model_api.id
  
  triggers = {
    # NOTE: This will ensure the deployment happens when resources change
    redeployment = sha1(jsonencode([
      aws_api_gateway_resource.models.id,
      aws_api_gateway_method.upload.id,
      aws_api_gateway_integration.lambda.id,
      aws_api_gateway_resource.upload.id,
      aws_api_gateway_method.upload_post.id,
      aws_api_gateway_integration.upload_integration.id,
      timestamp()  # Force redeployment
    ]))
  }
  
  lifecycle {
    create_before_destroy = true
  }
}

# Then create stage
resource "aws_api_gateway_stage" "api" {
  deployment_id = aws_api_gateway_deployment.api.id
  rest_api_id   = aws_api_gateway_rest_api.model_api.id
  stage_name    = "dev"
}

# Then create usage plan
resource "aws_api_gateway_usage_plan" "upload_plan" {
  name = "checkers-ai-upload-plan"

  api_stages {
    api_id = aws_api_gateway_rest_api.model_api.id
    stage  = aws_api_gateway_stage.api.stage_name
  }

  quota_settings {
    limit  = 100
    period = "DAY"
  }

  throttle_settings {
    burst_limit = 5
    rate_limit  = 10
  }

  depends_on = [aws_api_gateway_stage.api]
}

# Add API key to usage plan
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
          "s3:GetObject",
          "s3:PutObjectAcl",
          "s3:GetObjectAcl",
          "s3:ListBucket",
          "s3:GetBucketPolicy",
          "s3:PutBucketPolicy",
          "s3:GetBucketLocation",
          "dynamodb:PutItem",
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = [
          "${aws_s3_bucket.model_storage.arn}/*",
          aws_s3_bucket.model_storage.arn,
          aws_dynamodb_table.model_metadata.arn,
          "arn:aws:logs:*:*:*"
        ]
      }
    ]
  })
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

# Add new API Gateway resource for upload URLs
resource "aws_api_gateway_resource" "upload_urls" {
  rest_api_id = aws_api_gateway_rest_api.model_api.id
  parent_id   = aws_api_gateway_resource.models.id
  path_part   = "upload-urls"
}

# Add GET method for upload URLs
resource "aws_api_gateway_method" "get_upload_urls" {
  rest_api_id   = aws_api_gateway_rest_api.model_api.id
  resource_id   = aws_api_gateway_resource.upload_urls.id
  http_method   = "GET"
  authorization = "NONE"
  api_key_required = true
}

# Add integration for upload URLs
resource "aws_api_gateway_integration" "get_upload_urls" {
  rest_api_id = aws_api_gateway_rest_api.model_api.id
  resource_id = aws_api_gateway_resource.upload_urls.id
  http_method = aws_api_gateway_method.get_upload_urls.http_method
  
  integration_http_method = "POST"
  type                   = "AWS_PROXY"
  uri                    = aws_lambda_function.upload_handler.invoke_arn
}

# Add presigned URL endpoint
resource "aws_api_gateway_resource" "presigned_url" {
  rest_api_id = aws_api_gateway_rest_api.model_api.id
  parent_id   = aws_api_gateway_resource.models.id
  path_part   = "presigned-url"
}

resource "aws_api_gateway_method" "get_presigned_url" {
  rest_api_id      = aws_api_gateway_rest_api.model_api.id
  resource_id      = aws_api_gateway_resource.presigned_url.id
  http_method      = "GET"
  authorization    = "NONE"
  api_key_required = true

  request_parameters = {
    "method.request.header.x-api-key" = true
  }
}

resource "aws_api_gateway_integration" "presigned_url" {
  rest_api_id             = aws_api_gateway_rest_api.model_api.id
  resource_id             = aws_api_gateway_resource.presigned_url.id
  http_method             = aws_api_gateway_method.get_presigned_url.http_method
  integration_http_method = "POST"
  type                    = "AWS_PROXY"
  uri                     = aws_lambda_function.presigned_url_handler.invoke_arn
  
  request_parameters = {
    "integration.request.header.x-api-key" = "method.request.header.x-api-key"
  }
}

# Add Lambda for presigned URLs
resource "aws_lambda_function" "presigned_url_handler" {
  filename      = "presigned_url_handler.zip"
  function_name = "model-presigned-url-handler"
  role         = aws_iam_role.lambda_role.arn
  handler      = "index.handler"
  runtime      = "python3.9"
}

# Add Lambda permission for presigned URL handler
resource "aws_lambda_permission" "presigned_url" {
  statement_id  = "AllowAPIGatewayInvokePresignedUrl"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.presigned_url_handler.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_api_gateway_rest_api.model_api.execution_arn}/*/*"
}

# Add upload endpoint
resource "aws_api_gateway_resource" "upload" {
  rest_api_id = aws_api_gateway_rest_api.model_api.id
  parent_id   = aws_api_gateway_resource.models.id
  path_part   = "upload"
}

# POST method for upload endpoint
resource "aws_api_gateway_method" "upload_post" {
  rest_api_id      = aws_api_gateway_rest_api.model_api.id
  resource_id      = aws_api_gateway_resource.upload.id
  http_method      = "POST"
  authorization    = "NONE"
  api_key_required = true

  request_parameters = {
    "method.request.header.x-api-key" = true
  }
}

# Add method response
resource "aws_api_gateway_method_response" "upload_post" {
  rest_api_id = aws_api_gateway_rest_api.model_api.id
  resource_id = aws_api_gateway_resource.upload.id
  http_method = aws_api_gateway_method.upload_post.http_method
  status_code = "200"
}

# Integration for upload endpoint
resource "aws_api_gateway_integration" "upload_integration" {
  rest_api_id             = aws_api_gateway_rest_api.model_api.id
  resource_id             = aws_api_gateway_resource.upload.id
  http_method             = aws_api_gateway_method.upload_post.http_method
  integration_http_method = "POST"
  type                    = "AWS_PROXY"
  uri                     = aws_lambda_function.upload_handler.invoke_arn
}

# Add DynamoDB permissions to Lambda role
resource "aws_iam_role_policy_attachment" "lambda_dynamodb" {
  role       = aws_iam_role.lambda_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess"
}

output "api_key_id" {
  value = aws_api_gateway_api_key.upload_key.id
}

output "api_endpoint" {
  value = "${aws_api_gateway_deployment.api.invoke_url}${aws_api_gateway_stage.api.stage_name}/models"
}
