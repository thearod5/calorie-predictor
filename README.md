# Calorie Predictor
## Project Code 
### Running
In order to run the following experiment:

1. Create a virtual environment named `venv` and install requirements via `requirements.txt`
- `$ venv/bin/pip3 install -r requirements.txt`
2. Download all the data
3. Pre-process the data using the script under: `scripts/preprocessing/runner.py`
4. Modify the `constants.py` under the `dev` environment to point to the processed data
5. To run a given architecture on a task, use the bash script as follows:
- `$ ./runner.sh dev [TaskName] [ModelName]` where task name is calories, mass, ingredients
and model name is vgg, resnet, and xception. Models are saved via checkpoints, choosing the best performing epoch on the validation data.

### Folders
- cleaning: Responsible for parsing raw datasets into formatted objects
- data: Downloaded datasets.
- experiment: Experiment pipeline for pre-training and training.
- model: Contains the NN architecture
- results: Contains the metrics scores of the different models analzyed. 

-----------------------------------------------------------------------

## Report

### Introduction
In the following experiment, we set out to predict the number of calories in a food item from an image of it. The number of calories in food is primarily dependent on two factors: the type of food and the volume of the food. While datasets containing images of food are abundant, there are relatively few available datasets which also include volume information, and most of these contain well under 10k images. Furthermore, predicting volume from an image is challenging due to the different conditions (e.g. lighting, angles, depth) at which a photo may be taken. 

To overcome this problem, we plan to combine two pre-trained models. The first will classify food items and the second will predict the volume of food in a picture. Finally, we will combine these two models and fine-tune on the  calorie prediction task. These ensemble model will be compared against a series of baseline models trained purely at predicting the number of calories.

### Data
In total, we use five different datasets containing images of food which are described in Table 1 below. For each task, at least two datasets are selected and split to construct training, validation, and testing splits. A breakdown of which datasets will be used for each task can be seen in Table 2. With the exception of the food classification task, we use one of the datasets for training and validation and the other for testing. Due to the abundance of data for food classification, we also use a different dataset for validation in this task.

#### Table 1
| Name        | Description | Size        |
| ----------- | ----------- | ----------- |
| [Food Image Classification Data](https://kaggle.com/kmader/food41) | Images of food spanning over 101 categories.| 101 K |
| [UNIMB2016](http://www.ivl.disco.unimib.it/activities/food-recognition/) | Tray images with multiple foods and containing 15 food categories.| 2K |
| [ECUST Food Dataset](https://github.com/Liang-yc/ECUSTFD-resized-) | Images of food and their weight| 3K |
| [Nutrition 5k](https://github.com/google-research-datasets/Nutrition5k#download-data) | Images of food, their calorie counts, ingredients and weight| 5K |
| [MenuMatch](http://neelj.com/projects/menumatch/data/) | Images of food and their caloriecounts| 646 |



