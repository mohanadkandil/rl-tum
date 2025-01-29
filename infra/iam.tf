# Add API Gateway and Lambda permissions to your user
resource "aws_iam_user_policy" "api_gateway_access" {
  name = "api-gateway-lambda-access"
  user = "tum-rl"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "apigateway:*",
          "lambda:*",
          "iam:PassRole",
          "iam:CreateRole",
          "iam:PutRolePolicy"
        ]
        Resource = "*"
      }
    ]
  })
} 