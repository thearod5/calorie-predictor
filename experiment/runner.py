import os
import sys
# makes this runnable from command line
from typing import Dict

import numpy as np

from experiment.Food2Index import Food2Index

path_to_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(path_to_src)

import warnings
from enum import Enum

from constants import N_EPOCHS, set_env

from experiment.tasks.calorie_task import CaloriePredictionTask
from experiment.tasks.category_task import FoodClassificationTask
from experiment.tasks.mass_task import MassPredictionTask
from experiment.tasks.base_task import BaseModel, Task
from experiment.tasks.test_task import TestTask
import tensorflow as tf

warnings.filterwarnings("ignore")


class Tasks(Enum):
    TEST = TestTask
    CALORIE = CaloriePredictionTask
    MASS = MassPredictionTask
    INGREDIENTS = FoodClassificationTask


name2task: Dict[str, Tasks] = {
    "calories": Tasks.CALORIE,
    "mass": Tasks.MASS,
    "ingredients": Tasks.INGREDIENTS
}

name2model = {
    "vgg": BaseModel.VGG,
    "resnet": BaseModel.RESNET,
    "xception": BaseModel.XCEPTION,
    "test": BaseModel.TEST
}


def print_sample(dataset: tf.data.Dataset):
    food2index = Food2Index()
    food2count = {}
    for batch_images, batch_labels in dataset:
        for batch_index in range(batch_images.shape[0]):
            label = batch_labels[batch_index]
            label_indices = np.where(label == 1)[0]
            # image = batch_images[batch_index]
            # plt.imshow(np.asarray(image).astype(np.uint8))
            # plt.show()
            foods = []
            for label_index in label_indices:
                food_name = food2index.get_ingredient(label_index)
                foods.append(food_name)
                if food_name in food2count:
                    food2count[food_name] += 1
                else:
                    food2count[food_name] = 1

    sorted_food_counts = {k: v for k, v in sorted(food2count.items(), key=lambda item: item[1])}
    print("Foods Counts:", sorted_food_counts)


if __name__ == "__main__":
    # 1. Process arguments
    if len(sys.argv) != 4:
        raise Exception("Expected: [Env] [TaskName] [ModelName]")
    env, task_name, model = sys.argv[1:]

    # 2. Extract task and model
    set_env(env)
    task_selected: Tasks = name2task[task_name]

    base_model = name2model[model]

    # 3. Create task resources and train.
    task: Task = task_selected.value(base_model, n_epochs=N_EPOCHS)
    task.eval()
