#!/bin/bash

# Export AWS credentials
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"

# Submit job
sbatch train.slurm 