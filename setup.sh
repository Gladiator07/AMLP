#!/bin/bash

echo "This script will set your kaggle api-key to Kaggle api and download data for image problems"

echo "Installing dependencies"
pip3 install --upgrade --force-reinstall --no-deps kaggle

# Put your Kaggle api key path here
echo "Fetching your Kaggle API Key"
kaggle_api_key_path='/content/drive/MyDrive/Kaggle/kaggle.json'

# This snippet will install kaggle api and connect your api-key to it
mkdir -p ~/.kaggle
echo "Setting up your Kaggle key to API..."
cp $kaggle_api_key_path ~/.kaggle/
cat ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
echo "Kaggle API Key successfully linked !!!"

# This snippet will download the data in specified folder

# Specify the data path here
# data_path="/content/AMLP/Image_problem/Classification/input"
cd /content/AMLP/Image_problem/Classification/input_image
kaggle datasets download -d abhishek/siim-png-images
unzip siim-png-images
rm siim-png-images.zip 
rm -rf input/