variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "checkers-ai"
}

variable "environment" {
  description = "Environment (dev/prod)"
  type        = string
  default     = "dev"
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "tags" {
  description = "Default tags for all resources"
  type        = map(string)
  default = {
    Project     = "checkers-ai"
    Environment = "dev"
    ManagedBy   = "terraform"
  }
} 