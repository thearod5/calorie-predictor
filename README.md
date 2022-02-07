# Calorie Predictor
A research effort into predicting the number of calories from a picture of food.

- [ ] Download data and upload to data folder
- [ ] Combined food classification pre-training task data
- [ ] Combined food volume pre-training task data
- [ ] Combine prediction task data
- [ ] Create experiment pipeline
- [ ] Run experiment on servers and collect results

# Proposal Feedback
- [ ] Do not combine the datasets and then split into train/test/validation. Instead use each dataset for the type of split.
- [ ] Vary the order of the datasets used in the splits and collect interval error estimates.
- [ ] Do not use the softmax layer on a multi-classification problem.
- [ ] Use a non-linear regressor on top of the pre-trained model to come up with the final calorie estimate.
- [ ] Start the base models with their weights from ImageNet.
- [ ] ResNet architecture picture is actually DenseNet, whoops.

## Remaining Questions
- [ ] Verify that our metrics actually work for a multi-class classification problem.


# Folders
- cleaning: Responsible for parsing raw datasets into formatted objects
- data: Downloaded datasets.
- experiment: Experiment pipeline for pre-training and training.
- model: Contains the NN architecture
- results: Contains the metrics scores of the different models analzyed. 
