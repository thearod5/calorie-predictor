# Calorie Predictor

## FALL 2022 

### Conceptual Design
The following repository contains an experiment for improving the accuracy of a calorie prediction from the image of food.
This computer vision task attempts to use pre-training tasks to improve the mean absolute error on the test
data. Due to time constraints,only one pretraining step - mass predicition - was previously used, but the model will now include an additional pretraining step to learn to classify types of foods from an image. Furthermore, image segmentation will be used to guide the model in distinguishing ingredients on a plate separately, and we will apply data agumentation to improve generalizations on unseen food images. In addition, the project will leverage human insight by crowdsourcing which foods in an image appear most caloric to human viewers. This insight will be used to develop human-annotated saliency mapped into a loss function, using the [CYBORG framework](https://arxiv.org/abs/2112.00686). 

#### Data
In total, we use five different datasets containing images of food which are described in Table 1 below. For each task, at least two datasets are selected and split to construct training, validation, and testing splits. A breakdown of which datasets will be used for each task can be seen in Table 2. With the exception of the food classification task, we use one of the datasets for training and validation and the other for testing. Due to the abundance of data for food classification, we also use a different dataset for validation in this task. We will additionally collect a dataset from Amazon Turk which includes information about which foods in an image appear most caloric to human viewers. This dataset will be used to create saliency maps as discussed in the section above.

##### TABLE 1: Datasets used during the pre-training and regular training of our neural models.
| Name        | Description | Size        | Sample Characteristics |
| ----------- | ----------- | ----------- | ----------- |
| [Food-101](https://kaggle.com/kmader/food41) | Images of food spanning over 101 categories.| 101K | FOOD101-CHARACTERISTICS |
| [UNIMB2016](http://www.ivl.disco.unimib.it/activities/food-recognition/) | Tray images with multiple foods and containing 15 food categories.| 2K | CHARACTERISTICS |
| [ECUST Food Dataset](https://github.com/Liang-yc/ECUSTFD-resized-) | Images of food and their weight| 3K | ECUST-CHARACTERISTICS |
| [Nutrition 5k](https://github.com/google-research-datasets/Nutrition5k#download-data) | Images of food, their calorie counts, ingredients and weight| 5K | CHARACTERISTICS |
| [MenuMatch](http://neelj.com/projects/menumatch/data/) | Images of food and their calorie counts| 646 | MenuMatch-CHARACTERISTICS |
| [MenuMatch with human annotations]() | For each image in this dataset, we are hiring Amazon Mechanical Turk workers to create a bounding box over the region of the food they think is most caloric. See below for example. | 1938 | MenuMatch-Annotations-CHARACTERISTICS |

<p align="center">
  <img src="https://user-images.githubusercontent.com/26884108/192034289-e5ad072d-477e-4b88-9cd6-676b0c97dc14.jpg" alt="Example image for MenuMatch with human annotation."/>
</p>

##### TABLE 2: Tasks and the datasets used in each.
| Name        | Training Data | Validation Data | Testing Data |
| ----------- | -----------   | -----------     | ----------- |
| Food Classification (Pre-training) | [Food-101](https://kaggle.com/kmader/food41); [UNIMB2016](http://www.ivl.disco.unimib.it/activities/food-recognition/) | [Food-101](https://kaggle.com/kmader/food41); [UNIMB2016](http://www.ivl.disco.unimib.it/activities/food-recognition/) | N/A |
| Mass Prediction (Pre-training) | [Nutrition 5k](https://github.com/google-research-datasets/Nutrition5k#download-data) | [Nutrition 5k](https://github.com/google-research-datasets/Nutrition5k#download-data) | [ECUST Food Dataset](https://github.com/Liang-yc/ECUSTFD-resized-) |
| Calorie Prediction (Regular training) | [MenuMatch + Annotations](http://neelj.com/projects/menumatch/data/) | [MenuMatch + Annotations](http://neelj.com/projects/menumatch/data/) | [Nutrition 5k](https://github.com/google-research-datasets/Nutrition5k#download-data) |

## WINTER 2022

### Report

#### Introduction
In the following experiment, we set out to predict the number of calories in a food item from an image of it. The number of calories in food is primarily dependent on two factors: the type of food and the volume of the food. While datasets containing images of food are abundant, there are relatively few available datasets which also include volume information, and most of these contain well under 10k images. Furthermore, predicting volume from an image is challenging due to the different conditions (e.g. lighting, angles, depth) at which a photo may be taken. 

To overcome this problem, we plan to combine two pre-trained models. The first will classify food items and the second will predict the volume of food in a picture. Finally, we will combine these two models and fine-tune on the  calorie prediction task. These ensemble model will be compared against a series of baseline models trained purely at predicting the number of calories.

#### Data
In total, we use five different datasets containing images of food which are described in Table 1 below. For each task, at least two datasets are selected and split to construct training, validation, and testing splits. A breakdown of which datasets will be used for each task can be seen in Table 2. With the exception of the food classification task, we use one of the datasets for training and validation and the other for testing. Due to the abundance of data for food classification, we also use a different dataset for validation in this task.

#### Neural Models
***VGG16:*** Originating from the *Oxford Visual Geometry Group*, this model uses a series of small convolutional filters (3x3) stacked deeply in a series of 16 layers. Specifically, the model accepts an RGB image of size 224x224 and passes it through 2 layers of 64 3x3 convolutions which are pooled using a max pooling layer. Then, this pattern of using 3x3 convolutions and pooling is repeated with the number of convolutions being 128, 256, 512, and 512. Finally, there three connected layers lie at the end of the convolutional layers containing 4096, 4096, and 1000 with the softmax activation function applied to transforms the final values into a prediction of 1000 classes.

***ResNET:*** The residual network model, otherwise known as ResNet, is deep convolutional model based from the VGG architecture but which leverages skip connects to jump over some layers -- mimicking the biology of our brain.

***Xception:***  Xception is a convolution neural model based on depthwise separable convolution layers. This model relies on the assumption that the mapping of cross-channels correlations and spatial correlations in the feature maps of convolutional neural networks can be entirely decoupled. Xception is an more robust and efficient version of the Inception model.

Initially, we added only a dense layer to all models. However, due to poor performance in the classification task, we added several additional layers for the food classification models. These additional layers (in order) include an average pooling layer, a dense layer with relu activation, and a dropout layer. We keep the final dense layer for classification but apply the softmax activation function. For regression models, we keep just one dense layer with a single neuron for the prediction of mass or calorie amounts. 

***Ensemble:*** For the ensemble model, we choose the mass model that achieved the best results and remove the final prediction layer in-order to extract the features encoded by the model. This is fed as input into our ensemble model and passed through a dense hidden layer with 100 neurons and then through a final dense layer with a single neuron for the calorie prediction.  

#### Metrics

The first pre-training task focuses on predicting what categories of food are present in an image. Our predictions will be represented by an encoded vector of size *n* containing 1 wherever a food category is present and 0 otherwise where *n* is the number of food classes spanning our pre-training data. We plan to use the **categorical crossentropy** loss function along with **log loss** as our training metrics based on common uses in the field. We calculate the **accuracy** to evaluate the models.

Meanwhile, the second pre-training task (predicting volume) as well as our primary training task (predicting the number of calories) are both regression problems. Therefore, we are choosing to use the **Root Mean Squared Error** as our training metric and subsequently choose the **Mean Squared Error** loss function to train the models as recommended by other practitioners. We calculate the **Mean Absolute Error (MAE)** to evaluate the models.

#### Results
In Table 3, we present the results for the three tasks across the three neural network architectures beginning with the ImageNet weights. We also show the final ensemble model using the best model from the mass classification as pre-training.

The first task predicts the mass of the food, and the second task classifies which ingredients are in the pictures. Initially, all food classification accuracies were less than 10%. After adding additional layers and the softmax activation function, our accuracies improved substantially as shown in Table 3. However, because we had to retrain all our models, training was not done within the time frame necessary to use it in our pre-training step.
Therefore, we only use the mass prediction as a pre-training step for our final calorie predictor task, but we hope to also incorporate the food classification in the future. 

The final task predicts the number of calories in a series of images. We show the results with no other pre-training as our baseline performance measure as well as the final results with the mass prediction as a pre-training step. 

##### TABLE 3: Results on tasks.
| Task                            | Architecture | Result (Validation Data) | Result (Test Data)     | Metric      |
| ------------------------------- | ------------ | ------------------------ | ---------------------- | ----------- |
| Mass Prediction                 | VGG          | 71.6 g                   | N/A                    | MAE         |
| Mass Prediction                 | ResNet       | 68.7 g                   | N/A                    | MAE         |
| Mass Prediction                 | Xception     | 71.3 g                   | N/A                    | MAE         |
| Food Classification             | VGG          | In Progress              | N/A                    | Accuracy    |
| Food Classification             | ResNet       | In Progress              | N/A                    | Accuracy    |
| Food Classification             | Xception     | In Progress              | N/A                    | Accuracy    |
| Calorie Prediction (Baseline)   | VGG          | 102.1 cal                | **354.9 cal**          | MAE         |
| Calorie Prediction (Baseline)   | ResNet       | 91.9 cal                 | 386.4 cal              | MAE         |
| Calorie Prediction (Baseline)   | Xception     | 87.9 cal                 | 361.1 cal              | MAE         |
| Calorie Prediction              | * Ensemble   | 92.5 cal                 | 375.1 cal              | MAE         |

\* The ensemble model uses the ResNet mass prediction model as pre-training since it performed the best on the validation data.

#### Analysis
The ResNet architecture obtains the lowest MAE for the mass prediction while 
we await the results for the food classification. 
In our baseline calorie prediction model, we see that VGG performs best on the test data. 
Unfortunately, our ensemble model does not outperform VGG, but we do see a slight 
improvement compared to the ResNet model. Ultimately, it is clear that calorie 
prediction is a challenging task. Despite the relatively low MAE achieved in the 
mass prediction, the ensemble calorie prediction model has a MAE of 375 calories 
which would be a substantial difference to those interested in tracking calories. 
Nonetheless, we are not discouraged as we believe that the additional steps described 
in the subsequent section will serve to greatly improve this result.  

#### Future Work
In the future, we plan to augment our data with different rotations and croppings of our input images. 
Furthermore, we will use the food classification as another pre-training step to strengthen our calorie
 prediction ensemble model. Finally, we will experiment with adding additional layers to the ensemble 
 model as well as changing the number of neurons on in our hidden layer.   

## RUNNING 
In order to run the following experiment:
1. Setup development environment
2. Create production data
3. Run experiment via run script

**Setup Development Environment**
1. Create a virtual environment named `venv` and install requirements via `requirements.txt`
> $ venv/bin/pip3 install -r requirements.txt

**Creating Production Data**
1. Download each dataset present in the report below and place into a folder.
2. Update `PATH_TO_PROJECT` to point to this containing folder.
3. Run script `scripts/preprocessing/runner.py` to create production data.

**Run Experiment**
5. To run a given architecture on a task, use the bash script as follows:
- `$ ./runner.sh dev [data] [task] [model] [mode]` where:

data: test | prod
task: mass | ingredients | calories
model: vgg | resnet | xception
mode: train | eval

### Folders
- cleaning: Responsible for parsing raw datasets into formatted objects
- data: Downloaded datasets.
- experiment: Experiment pipeline for pre-training and training.
- logging_util: Logger configuration
- results: Model checkpoints while training
- scripts: Pre-processing scripts (for now).
