#!/bin/bash

# Create a temporary directory
mkdir -p lambda_package
cp ../presigned_url_handler.py lambda_package/index.py

# Create zip file
cd lambda_package
zip -r ../presigned_url_handler.zip .
cd ..

# Clean up
rm -rf lambda_package 