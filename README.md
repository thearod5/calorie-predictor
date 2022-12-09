# Calorie Predictor
The following repository implements a framework for crerating CNNs with specific task architectures and implements the [CYBORG](https://arxiv.org/abs/2112.00686) loss function for them in TensorFlow. This codebase was developed for researching the effect of leveraging human saliency within the task of calorie prediction. Currently available models include ResNet, Xception, and an definable ensemble model. Other models are easily implementable via a `ModelManager` (described below).

The experiment design, results, and future work of this research can be found [here](https://drive.google.com/file/d/16RJtji8drDsiTuDpCOk40XdLUyD9TKSd/view?usp=sharing).

# Getting Started 
## Setup
In order to run the following experiment:
1.  Create a virtual environment named `venv` and install requirements via `requirements.txt`
> $ venv/bin/pip3 install -r requirements.txt
2. Download production [data](https://calorie-predictor.s3.us-east-2.amazonaws.com/processed.zip) (see below for re-creating this data).
3. Create `.env` file containing:
> ENV=prod

## Running Training or Evaluation
 To run training or evaluation, navigate to the `src/scripts` folder within the project and run the following command ([field_name=default_value]):
> $ ./runner.sh dev [job] [task] [model_manager]

`job`: `train` | `eval` <br />
`state`: `new` | `load` <br />
`model_manager`: `resnet` | `xception` | `ensemble` <br /> <br />
Optional Parameters <br />
`--path`: Path to load model. Required if `state=load` <br />
`--export`: Path to export checkpoints to. Required if exporting to somewhere other than `--path`.<br />
`--nocam`: Enabling keras default train and evaulation loops. Otherwise the `CamTrainer` is used. <br />
`--task`: Defines the task to run, must be one of `mass` | `ingredients` | `calories`. Implement new task within class `Tasks`.  <br />
`--project`: The path to appended to `path` and `export`. This defaults to `results/checkpoints` within the repository. <br />
`--alpha`: Deprecated. Used to define the CYBORG loss combination function. Currently has not effect. <br />
`--usesplit`: Whether to trigger the formal train/test splits of the nutrition5k dataset for the task of calorie prediction. <br />

## Creating Production Data
1. Download each dataset present in the report below and place into a folder.
2. Update `PATH_TO_PROJECT` to point to this containing folder.
3. Run script `scripts/preprocessing/runner.py` to create production data.

# System Architecture

## Definitions
`CamTrainer`: The trainer implementing a custom loss function using human saliency maps.
`Dataset`: Defines where to find the data for a dataset and how to parse it.
`ModelManager`: Manages the a specific architecture for a model, including defining the creation of instances the details used for the CYBORG loss function (e.g. last convolutional layer).
`Tasks`: Defines a task specific information (e.g. number of outputs) for creating models for it.
`Processor`: Defines pre-processing steps necessary for reading a dataset.

## Folders
- `scripts`: Entry point into evaluation or training on tasks.
- `cleaning`: Responsible for parsing raw datasets into formatted objects
- `data`: Downloaded datasets.
- `experiment`: Tasks definitions for experiments including pre-training and training.
- `logging_util`: Logger configuration
- `results`: Model checkpoints while training
- `preprocessing`: Pre-processing scripts (for now).
