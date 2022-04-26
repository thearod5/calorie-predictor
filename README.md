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

#### TABLE 1: Datasets used during the pre-training and regular training of our neural models.
| Name        | Description | Size        |
| ----------- | ----------- | ----------- |
| [Food-101](https://kaggle.com/kmader/food41) | Images of food spanning over 101 categories.| 101K |
| [UNIMB2016](http://www.ivl.disco.unimib.it/activities/food-recognition/) | Tray images with multiple foods and containing 15 food categories.| 2K |
| [ECUST Food Dataset](https://github.com/Liang-yc/ECUSTFD-resized-) | Images of food and their weight| 3K |
| [Nutrition 5k](https://github.com/google-research-datasets/Nutrition5k#download-data) | Images of food, their calorie counts, ingredients and weight| 5K |
| [MenuMatch](http://neelj.com/projects/menumatch/data/) | Images of food and their caloriecounts| 646 |

#### TABLE 2: Tasks and the datasets used in each.
| Name        | Training Data | Validation Data | Testing Data |
| ----------- | -----------   | -----------     | ----------- |
| Food Classification (Pre-training) | [Food-101](https://kaggle.com/kmader/food41); [UNIMB2016](http://www.ivl.disco.unimib.it/activities/food-recognition/) | [Food-101](https://kaggle.com/kmader/food41); [UNIMB2016](http://www.ivl.disco.unimib.it/activities/food-recognition/) | N/A |
| Mass Prediction (Pre-training) | [Nutrition 5k](https://github.com/google-research-datasets/Nutrition5k#download-data) | [Nutrition 5k](https://github.com/google-research-datasets/Nutrition5k#download-data) | [ECUST Food Dataset](https://github.com/Liang-yc/ECUSTFD-resized-) |
| Calorie Prediction (Regular training) | [Nutrition 5k](https://github.com/google-research-datasets/Nutrition5k#download-data) | [Nutrition 5k](https://github.com/google-research-datasets/Nutrition5k#download-data) | [MenuMatch](http://neelj.com/projects/menumatch/data/) |

### Neural Models
***VGG16:*** Originating from the *Oxford Visual Geometry Group*, this model uses a series of small convolutional filters (3x3) stacked deeply in a series of 16 layers. Specifically, the model accepts an RGB image of size 224x224 and passes it through 2 layers of 64 3x3 convolutions which are pooled using a max pooling layer. Then, this pattern of using 3x3 convolutions and pooling is repeated with the number of convolutions being 128, 256, 512, and 512. Finally, there three connected layers lie at the end of the convolutional layers containing 4096, 4096, and 1000 with the softmax activation function applied to transforms the final values into a prediction of 1000 classes.

***ResNET:*** The residual network model, otherwise known as ResNet, is deep convolutional model based from the VGG architecture but which leverages skip connects to jump over some layers -- mimicking the biology of our brain.

***Xception:***  Xception is a convolution neural model based on depthwise separable convolution layers. This model relies on the assumption that the mapping of cross-channels correlations and spatial correlations in the feature maps of convolutional neural networks can be entirely decoupled. Xception is an more robust and efficient version of the Inception model.

Initially, we added only a dense layer to all models. However, due to poor performance in the classification task, we added several additional layers for the food classification models. These additional layers (in order) include an average pooling layer, a dense layer with relu activation, and a dropout layer. We keep the final dense layer for classification but apply the softmax activation function. For regression models, we keep just one dense layer with a single neuron for the prediction of mass or calorie amounts. 

### Metrics

The first pre-training task focuses on predicting what categories of food are present in an image. Our predictions will be represented by an encoded vector of size *n* containing 1 wherever a food category is present and 0 otherwise where *n* is the number of food classes spanning our pre-training data. We plan to use the **categorical crossentropy** loss function along with **log loss** as our training metrics based on common uses in the field. We calculate the **accuracy** to evaluate the models.

Meanwhile, the second pre-training task (predicting volume) as well as our primary training task (predicting the number of calories) are both regression problems. Therefore, we are choosing to use the **Root Mean Squared Error** as our training metric and subsequently choose the **Mean Squared Error** loss function to train the models as recommended by other practitioners. We calculate the **Mean Absolute Error (MAE)** to evaluate the models.

### Results
In Table 3, we present the results for the three tasks (plus the calorie baseline model) across the three neural network architectures beginning with the ImageNet weights.

The first task predicts the mass of the food, and the second task classifies which ingredients are in the pictures. Initially, all food classification accuracies were less than 10%. After adding additional layers and the softmax activation function, our accuracies improved substantially as shown in Table 3. However, because we had to retrain all our models, training was not done within the time frame necessary to use it in our pretraining step.
Therefore, we only use the mass prediction as a pre-training step for our final calorie predictor task, but we hope to also incorporate the food classification in the future. 

The final task predicts the number of calories in a series of images. We show the results with no other pre-training as our baseline performance measure as well as the final results with the mass prediction as a pre-training step. 

