#!/bin/bash

echo "This script will set your kaggle api-key to Kaggle api and download data for image problems"

echo "Installing dependencies"
pip3 install --upgrade --force-reinstall --no-deps kaggle
pip3 install pretrainedmodels

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
cd /content/AMLP/Image_Problem/Classification/
mkdir input_image
cd input_image/
kaggle datasets download -d abhishek/siim-png-images
kaggle datasets download -d abhishek/siim-png-train-csv
unzip siim-png-images
unzip siim-png-train-csv.zip 
rm siim-png-images.zip 
rm siim-png-train-csv.zip
rm -rf input/

echo "Creating pneumothorax images cv folds...."
folds_script="/content/AMLP/Image_Problem/Classification/src/create_folds.py"
python3 $folds_script

echo "Done. You are all set to work..."