# Calorie Predictor

## Summary
The following report describes an experiment examining the performance of popular CNNs on a the calorie prediction task. The baseline method uses ResNet, Xception, and other CNNS to train directly on this task. Our first experimental condition pre-trains a model on a food classification task before the downstream calorie prediction task. Lastly, we modify the [CYBORG](https://arxiv.org/abs/2112.00686) loss function for leveraging human annotated feature maps for enhacing the inner features of the models.

[Project Report](https://drive.google.com/file/d/16RJtji8drDsiTuDpCOk40XdLUyD9TKSd/view?usp=sharing) containing experiment design, results, and future work. Please use this if you are grading this project.

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
