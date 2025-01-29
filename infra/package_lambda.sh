#!/bin/bash

mkdir -p lambda_package
cp ../upload_handler.py lambda_package/index.py
d lambda_package
zip -r ../upload_handler.zip .
cd ..

rm -rf lambda_package 