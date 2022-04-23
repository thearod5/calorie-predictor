import os

import tensorflow as tf
from keras.models import load_model

from constants import PROJECT_DIR
from experiment.models.checkpoint_creator import get_checkpoint_path

if __name__ == "__main__":
    task_name = "CaloriePredictionTask"
    base_model_name = "vgg"
    model = load_model(get_checkpoint_path(task_name, base_model_name))
    model_json = model.to_json()
    model_export_path = os.path.join(PROJECT_DIR, "results", "example.json")
    with open(model_export_path, "w") as model_file:
        model_file.write(model_json)
    model_path = os.path.join(PROJECT_DIR, "results", "pretrained-classifier.json")
    model_file = open(model_path).read()
    model = tf.keras.models.model_from_json(model_file)
    model.summary()
    model_file.close()
