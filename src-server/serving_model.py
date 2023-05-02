#!env python
import base64
import datetime
import io
import json
import logging
import os
import shutil

import abc
import collections
import enum
import tarfile
import time
import functools

import botocore
import numpy as np
import tensorflow as tf
import sagemaker.tensorflow
import sagemaker.serverless
import sagemaker.serializers
from PIL import Image
from sagemaker import ModelPackage
from sagemaker.huggingface import HuggingFaceModel

import tensorflowmodels
import common

logging.basicConfig()
log = logging.getLogger("layercake")
log.setLevel(logging.INFO)


class ActionState(enum.Enum):
  LOAD = 1,
  UNLOAD = 2,
  INFER = 3,
  INFER_COLD = 4

class ImageB64Serializer(sagemaker.serializers.JSONSerializer):
  def serialize(self, data):
    return json.dumps({"instances": [ [base64.urlsafe_b64encode(data).decode('utf-8')] ]})



class Measurement(list):
  def avg(self):
    if len(self) == 0:
      return 0.0
    return np.mean(self)

  def stddev(self):
    if len(self) == 0:
      return 0.0
    return np.std(self)

  def __str__(self):
    return f"{self.avg():0.3f} +/- {self.stddev():0.3f}"


class Model(abc.ABC):
  def __init__(self, name, *args, **kwargs):
    self.name = name
    self.is_loaded = False  # todo: this will change depending on our application -- serverless is lower than dedicated

    self._coldexec_latency = 0.0  # Latency in seconds todo: override
    self._exec_latency = 0.0  # Cold latency in seconds todo: override

  @property
  def latency(self):
    if self.is_loaded:
      return self.exec_latency
    else:
      return self.exec_latency + self.coldexec_latency

  @abc.abstractmethod
  def prepare(self, *args, **kwargs):
    """Prepares the model for deployment by configuring an endpoint, but not actually deploying an endpoint"""
    pass

  @abc.abstractmethod
  def infer(self, data, *args, **kwargs):
    """Runs inference on given data"""
    pass

  @abc.abstractmethod
  def remove(self, *args, **kwargs):
    """Removes endpoint"""
    pass

  @abc.abstractmethod
  def cleanup(self, *args, **kwargs):
    pass


class SageMakerModel(object):
  bucket_name = "layercake.scratch"
  role = "sagemaker-layercake"
  sm_session = sagemaker.Session(default_bucket=bucket_name)
  deployment_modifier = datetime.datetime.now().strftime('%b%d_%I%M%p')

  # todo: make this so we file by usage or application
  tarball_dir_prefix = "model_tarballs"

  def __init__(self, name, sagemaker_model, dimensions, accuracy, application=common.Application.IMAGE, *args, **kwargs):
    super().__init__(*args, **kwargs)

    if self.__class__.sm_session is None:
      self.__class__.sm_session = sagemaker.Session(default_bucket=self.bucket_name)

    self.name = name
    self.sagemaker_model = sagemaker_model

    self.dimensions = dimensions
    self.accuracy = accuracy
    self.measurements = collections.defaultdict(Measurement)

    self.application = application
    log.debug(f"Creating model with application: {application}")

    self.use_serverless = True if "use_serverless" not in kwargs else kwargs["use_serverless"]
    self.instance_type = "ml.t2.medium" if "instance_type" not in kwargs else kwargs["instance_type"]

    self.first_run = True

  def __str__(self):
    return f"<{self.name}, {self.accuracy}>"

  @classmethod
  def get_SagemakerModel(cls, tf_model, name, accuracy, *args, **kwargs):
    bucket_tarball = cls.generate_model_tarball(tf_model, name)
    sagemaker_model = cls.create_sagemaker_model(bucket_tarball)
    return SageMakerModel(name, sagemaker_model, tf_model.input_shape[1:3], accuracy)

  @classmethod
  def get_SagemakerModelFromARN(cls, arn, name, accuracy, *args, **kwargs):
    sagemaker_model = ModelPackage(role=cls.role, model_package_arn=arn, sagemaker_session=cls.sm_session)
    return SageMakerModel(name, sagemaker_model, 0, 0.80)

  @classmethod
  def get_SagemakerModelFromHuggingFace(cls, name, accuracy, model_id='cardiffnlp/twitter-roberta-base-sentiment'):
    # https://aws.amazon.com/blogs/machine-learning/host-hugging-face-transformer-models-using-amazon-sagemaker-serverless-inference/
    # https://github.com/huggingface/notebooks/blob/main/sagemaker/19_serverless_inference/sagemaker-notebook.ipynb
    # https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english
    #image_uri='763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:1.9.1-transformers4.12.3-cpu-py38-ubuntu20.04'
    hub = {
      'HF_MODEL_ID':model_id,
      'HF_TASK':'text-classification'
    }
    huggingface_model = HuggingFaceModel(
      env=hub,                      # configuration for loading model from Hub
      role=cls.role,
      # iam role with permissions
      transformers_version="4.17",  # transformers version used
      pytorch_version="1.10",        # pytorch version used
      py_version='py38',            # python version used
      #image_uri=image_uri,          # image uri
    )
    return SageMakerModel(name, huggingface_model, (50,50), accuracy, application=common.Application.TEXT)


  @classmethod
  def generate_model_tarball(cls, tf_model, name):
    # todo: check to see if model already exists in bucket

    def preprocess_and_decode(img_str):
      img = tf.io.decode_base64(img_str)
      img = tf.image.decode_jpeg(img, channels=3)
      img = tf.cast(img, dtype=tf.float32)
      return img

    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape = (1,),dtype="string"))
    model.add(tf.keras.layers.Lambda(lambda img : tf.map_fn(lambda im : preprocess_and_decode(im[0]), img, dtype="float32")))

    model.add(tf_model)
    tf_model = model

    # Write model out locally
    tf_model.save(f"{name}/1")

    # Zip model
    with tarfile.open(f"{name}.tar.gz", "w:gz") as tar:
      tar.add(f"{name}")

    # Upload model
    s3_response = cls.sm_session.upload_data(f"{name}.tar.gz",
                                             bucket=cls.bucket_name,
                                             key_prefix=f"{cls.tarball_dir_prefix}")
    shutil.rmtree(f"{name}")
    os.remove(f"{name}.tar.gz")
    return f"s3://{cls.bucket_name}/{cls.tarball_dir_prefix}/{name}.tar.gz"

  @classmethod
  def create_sagemaker_model(cls, bucket_tarball):
    # https://sagemaker.readthedocs.io/en/stable/api/utility/session.html#sagemaker.session.Session.create_model
    return sagemaker.tensorflow.TensorFlowModel(
      model_data=bucket_tarball,
      role=cls.role,
      framework_version="2.3",
    )

  @classmethod
  def setup_available_models(cls, *args, **kwargs):
    """Sets up all available models from tensorflowmodels.py to be served"""
    deployed_models = SageMakerModelEndpoint.get_active_endpoints()
    for m in deployed_models:
      log.info(f"Found an already loaded endpoint: {m}")

    models_to_setup = tensorflowmodels.get_models(list(deployed_models.keys()))
    sagemaker_models = [
      cls.get_SagemakerModel(m, m_name, accuracy)
      for m, m_name, accuracy in models_to_setup
    ]

    # https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad
    sagemaker_models.append(cls.get_SagemakerModelFromHuggingFace("BERT-bert-large-uncased-whole-word-masking-finetuned-squad", 0.8691, "bert-large-uncased-whole-word-masking-finetuned-squad"))

    for m in sagemaker_models:
      if m.name in deployed_models:
        # Then we already have an active deployment
        log.info(f"Skipping {m.name} since it is already deployed")
        continue
      deployed_model = m.deploy(*args, **kwargs)
      deployed_models[deployed_model.name] = deployed_model

    return deployed_models


  def deploy(self, *args, **kwargs):
    log.info(f"Deploy: {self}")
    if self.use_serverless:
      serverless_config = sagemaker.serverless.ServerlessInferenceConfig(
        memory_size_in_mb=6144,
        max_concurrency=1,
      )
      self.endpoint = self.sagemaker_model.deploy(
        serverless_inference_config=serverless_config,
        serializer=(ImageB64Serializer() if self.application == common.Application.IMAGE else sagemaker.serializers.JSONSerializer())
      )
    else:
      self.endpoint = self.sagemaker_model.deploy(
        initial_instance_count=1,
        instance_type=self.instance_type,
        serializer=ImageB64Serializer()
      )
    self.is_deployed = True
    log.info(f"Deployed {self.name} as {self.endpoint.endpoint_name}")
    kind_of_endpoint = (SageMakerModelEndpoint_Image
                        if self.application == common.Application.IMAGE
                        else SageMakerModelEndpoint_Text)
    log.debug(f"Returning an enpoind of kind: {kind_of_endpoint}")
    return kind_of_endpoint(self.name, self.endpoint, self.dimensions, self.accuracy, *args, **kwargs)


class SageMakerModelEndpoint(Model):

  _all_endpoints = set([])
  endpoint_info_file = "endpoints.info"

  def __init__(self, name, endpoint, dimensions, accuracy, *args, **kwargs):
    super().__init__(name, *args, **kwargs)
    self.name = name
    self.endpoint = endpoint
    self.dimensions = dimensions
    self._accuracy = accuracy
    self.__class__._all_endpoints.add(self)

    self.measurements = collections.defaultdict(Measurement)
    self.first_run = True


  def __str__(self):
    return f"<{self.name}, {self.accuracy} {self.exec_latency:0.3f}s ({self.load_latency:0.3f})>"

  @property
  @abc.abstractmethod
  def application(self):
    pass

  @classmethod
  def get_active_endpoints(cls):
    deployed_models = {}
    if os.path.exists(cls.endpoint_info_file):
      # then we load existing endpoints from there
      with open(cls.endpoint_info_file) as fid:
        deployment_lines = [l.strip() for l in fid.readlines()]
      for line in deployment_lines:
        endpoint = cls.from_json(line)
        deployed_models[endpoint.name] = endpoint
    return deployed_models

  @classmethod
  def document_active_endpoints(cls):
    with open(cls.endpoint_info_file, 'w') as fid:
      for endpoint in sorted(cls._all_endpoints, key=(lambda e: e.name)):
        fid.write(endpoint.to_json())
        fid.write('\n')

  def to_json(self):
    return json.dumps({
      "name" : self.name,
      "endpoint_name" : self.endpoint.endpoint_name,
      "dimensions" : list(self.dimensions),
      "accuracy" : self.accuracy,
      "application" : self.application
    })

  @classmethod
  def from_json(cls, endpoint_json):
    endpoint_dict = json.loads(endpoint_json)
    if "application" not in endpoint_dict or endpoint_dict["application"] == common.Application.IMAGE:
      new_cls = SageMakerModelEndpoint_Image
      endpoint = sagemaker.predictor.Predictor(endpoint_name=endpoint_dict["endpoint_name"],serializer=ImageB64Serializer())
    else:
      new_cls = SageMakerModelEndpoint_Text
      endpoint = sagemaker.predictor.Predictor(endpoint_name=endpoint_dict["endpoint_name"],serializer=sagemaker.serializers.JSONSerializer())
    return new_cls(
      name = endpoint_dict["name"],
      endpoint = endpoint,
      dimensions = tuple(endpoint_dict["dimensions"]), # todo: parse to and from json appropriate
      accuracy = endpoint_dict["accuracy"],
      #type = application
    )

  def record_time(action: ActionState):
    def record_decorator(fn):
      @functools.wraps(fn)
      def _impl(self, *fn_args, **fn_kwargs):
        ts = time.time()
        response = fn(self, *fn_args, **fn_kwargs)
        te = time.time()
        if self.first_run and action == ActionState.INFER:
          # action = ActionState.INFER_COLD
          self.measurements[ActionState.INFER_COLD].append((te - ts))
          self.first_run = False
          self.is_loaded = True
        else:
          self.measurements[action].append((te - ts))
        if action == ActionState.UNLOAD:
          self.first_run = True
        return response

      return _impl

    return record_decorator

  ##############
  # Properties #
  ##############
  @property
  def exec_latency(self):
    return self.measurements[ActionState.INFER].avg()

  @property
  def coldexec_latency(self):
    return self.measurements[ActionState.INFER_COLD].avg()

  @property
  def load_latency(self):
    return self.measurements[ActionState.LOAD].avg()

  @property
  def unload_latency(self):
    return self.measurements[ActionState.UNLOAD].avg()

  @property
  def expected_latency(self):
    if self.is_loaded:
      return self.exec_latency
    else:
      return self.coldexec_latency

  @property
  def accuracy(self):
    return self._accuracy # todo: currently only works for efficientnets

  ##################
  # Common Methods #
  ##################
  def prepare(self, *args, **kwargs):
    if self.prepared:
      return
    self.generate_model_tarball()
    try:
      self.create_sagemaker_model()
    except botocore.exceptions.ClientError:
      self.generate_model_tarball()
      self.create_sagemaker_model()
    self.prepared = True

  @record_time(ActionState.INFER)
  def infer(self, data=None, do_resize=True, *args, **kwargs):
    self._infer(data, do_resize, *args, **kwargs)

  @abc.abstractmethod
  def _infer(self, data=None, do_resize=True, *args, **kwargs):
    pass

  @record_time(ActionState.UNLOAD)
  def cooldown(self, *args, **kwargs):
    pass

  def remove(self, *args, **kwargs):
    #return
    self.endpoint.delete_endpoint()
    self.is_deployed = False

  def cleanup(self, *args, **kwargs):
    # https://sagemaker.readthedocs.io/en/stable/api/utility/session.html#sagemaker.session.Session.delete_model
    self.sm_session.delete_model(self.sagemaker_model.name)

  ##################
  # Helper Methods #
  ##################
  def get_stats(self):
    return {
      action: self.measurements[action]
      for action in ActionState
    }

  def cycle_model(self, num_executions=20):
    #self.prepare()
    #self.deploy()
    self.infer()  # cold start
    for _ in range(num_executions):
      self.infer()  # real inference
    self.cooldown()
    #self.remove()


class SageMakerModelEndpoint_Image(SageMakerModelEndpoint):
  @property
  def application(self):
    return common.Application.IMAGE

  def _infer(self, data=None, do_resize=True, *args, **kwargs):
    ts = time.time()
    img_byte_arr = io.BytesIO()
    if (data is None):
      log.info("Using mug as input")
      img = Image.open("mug.jpg")
    else:
      log.info("Using gRPC as input")
      img = data
    tl = time.time()
    # todo: only do resize if not running in infaas mode
    if do_resize:
      img.resize(self.dimensions).save(img_byte_arr, format="JPEG")
    else:
      img_byte_arr = img
      # img.save(img_byte_arr, format="JPEG")
    ti = time.time()
    response = self.endpoint.predict(img_byte_arr.getvalue())
    if response is not dict:
      response = json.loads(response)
    prediction = np.argmax(response["predictions"], axis=1)
    te = time.time()
    log.info(f"{te - ts:0.3f} ({tl - ts:0.3f} + {ti - tl:0.3f} + {te - ti:0.3f}) ({self.dimensions})")
    return str(prediction)



class SageMakerModelEndpoint_Text(SageMakerModelEndpoint):
  @property
  def application(self):
    return common.Application.TEXT

  def _infer(self, data=None, do_resize=True, *args, **kwargs):
    response = self.endpoint.predict({
      'inputs': json.dumps({
        "question": "What's my name?",
        "context": "My name is Clara and I live in Berkeley.  My name is Clara and I live in Berkeley.  My name is "
                   "Clara and I live in Berkeley.  My name is Clara and I live in Berkeley.  My name is Clara and I "
                   "live in Berkeley.  My name is Clara and I live in Berkeley. "
      })
    })
    log.info(f"response: {response}")
    return response

def cycle_model(model, num_inferences=2, serverless=False):
  try:
    model.start_model(serverless=serverless)
    for _ in range(num_inferences):
      measurement, response = model.request_inference()
  finally:
    model.cleanup()
    pass

if __name__ == "__main__":
  for serverless in [True, False]:
    for model, model_name in tensorflowmodels.get_models():

      model = Model(model_name, model)
      print(f"{model_name} : {'serverless' if serverless else 'dedicated'}")
      for _ in range(3):
        cycle_model(model, serverless=serverless)
      for key, val in model.get_stats().items():
        print(f"{key} : {val}")
