import tensorflow as tf

from constants import INPUT_SIZE

vgg_19 = tf.keras.applications.VGG19(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=INPUT_SIZE,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)
