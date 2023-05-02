#!env python

import tensorflow as tf
import typing

tf.keras.backend.set_image_data_format("channels_last")
vgg19_model = tf.keras.applications.VGG19

nasnetmobile_model = tf.keras.applications.NASNetMobile

resnet152v2_model = tf.keras.applications.ResNet152V2
tf.keras.backend.set_image_data_format("channels_last")
efficientnetb0_model = tf.keras.applications.EfficientNetB0
efficientnetb1_model = tf.keras.applications.EfficientNetB1
efficientnetb2_model = tf.keras.applications.EfficientNetB2
efficientnetb3_model = tf.keras.applications.EfficientNetB3
efficientnetb4_model = tf.keras.applications.EfficientNetB4
efficientnetb5_model = tf.keras.applications.EfficientNetB5
efficientnetb6_model = tf.keras.applications.EfficientNetB6
efficientnetb7_model = tf.keras.applications.EfficientNetB7

def get_models(models_to_skip=[]) -> list[tuple[tf.keras.models, str, float]]:
  models = [
    (efficientnetb0_model, "efficientnetb0", 0.71),
    (efficientnetb1_model, "efficientnetb1", 0.791),
    (efficientnetb2_model, "efficientnetb2", 0.801),
    (efficientnetb3_model, "efficientnetb3", 0.816),
    (efficientnetb4_model, "efficientnetb4", 0.829),
    (efficientnetb5_model, "efficientnetb5", 0.836),
    (efficientnetb6_model, "efficientnetb6", 0.840),
    (efficientnetb7_model, "efficientnetb7", 0.843),
  ]
  return [(t[0](), t[1], t[2]) for t in models if t[1] not in models_to_skip]

