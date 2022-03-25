#!/bin/bash
#$ -M vhsalbertorodriguez@gmail.com
#$ -m abe

module load python

pip install --user virtualenv

~/.local/bin/virtualenv calorie-predictor

source calorie-predictor/bin/activate

pip install -r requirements.txt

export PATH=${HOME}/.local/bin:${PATH}

python experiment/runner.py calories vgg