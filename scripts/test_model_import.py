import os

import tensorflow as tf

from constants import PROJECT_DIR

if __name__ == "__main__":
    model_path = os.path.join(PROJECT_DIR, "results", "pretrained-classifier.json")
    model_file = open(model_path).read()
    model = tf.keras.models.model_from_json(model_file)
    model.summary()
    model_file.close()
