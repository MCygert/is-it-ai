#!/bin/bash

dir=$PWD
root_directory="is-it-ai"
dir="${dir##*/}"

if [ $dir != $root_directory ] 
    then
    echo 'You are not in root directory'
else
    if [ ! -d data ]; then
        mkdir data
    fi
    cd data
    kaggle competitions download -c llm-detect-ai-generated-text
    kaggle datasets download -d thedrcat/daigt-v2-train-dataset
    unzip llm-detect-ai-generated-text.zip 
    unzip daigt-v2-train-dataset.zip 
    rm llm-detect-ai-generated-text.zip
    rm daigt-v2-train-dataset.zip

fi
