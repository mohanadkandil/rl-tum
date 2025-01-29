terraform {
  backend "s3" {
    bucket         = "checkers-ai-terraform-state"
    key            = "terraform.tfstate"
    region         = "eu-central-1"
    dynamodb_table = "checkers-ai-terraform-locks"
    encrypt        = true
  }
} 