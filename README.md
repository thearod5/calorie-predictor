# Calorie Predictor
A research effort into predicting the number of calories from a picture of food.

# Running
In order to run the following experiment:

1. Create a virtual environment named `venv` and install requirements via `requirements.txt`
- `$ venv/bin/pip3 install -r requirements.txt`
2. Download all the data
3. Pre-process the data using the script under: `scripts/preprocessing/runner.py`
4. Modify the `constants.py` under the `dev` environment to point to the processed data
5. To run a given architecture on a task, use the bash script as follows:
- `$ ./runner.sh dev [TaskName] [ModelName]` where task name is calories, mass, ingredients
and model name is vgg, resnet, and xception. Models are saved via checkpoints, choosing the best performing epoch on the validation data.
 
# Proposal Feedback
- [x] Do not combine the datasets and then split into train/test/validation. Instead use each dataset for the type of split.
- [ ] Vary the order of the datasets used in the splits and collect interval error estimates.
- [ ] Do not use the softmax layer on a multi-classification problem.
- [ ] Use a non-linear regressor on top of the pre-trained model to come up with the final calorie estimate.
- [ ] Start the base models with their weights from ImageNet.
- [ ] ResNet architecture picture is actually DenseNet, whoops.

# Folders
- cleaning: Responsible for parsing raw datasets into formatted objects
- data: Downloaded datasets.
- experiment: Experiment pipeline for pre-training and training.
- model: Contains the NN architecture
- results: Contains the metrics scores of the different models analzyed. 
