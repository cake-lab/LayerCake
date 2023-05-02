#!env python
import functools
import logging
logging.basicConfig()
log = logging.getLogger("layercake")
#log = logging.getLogger()
log.setLevel(logging.DEBUG)

import abc

import serving_model
import exceptions

class ModelPicker(abc.ABC):
  def __init__(self, available_models):
    self.models = { model.name : model for model in available_models }

  @abc.abstractmethod
  def pick_model(self, *args, **kwargs):
    pass


class SimpleModelPicker(ModelPicker):
  def pick_model(self, model_name, *args, **kwargs):
    try:
      return self.models[model_name]
    except KeyError:
      return None

class INFaaSModelPicker(ModelPicker):

  class Decorators(object):
    @classmethod
    def mark_model_active(cls, decorated_fn):
      @functools.wraps(decorated_fn)
      def _impl(self, *fn_args, **fn_kwargs):
        response = decorated_fn(self, *fn_args, **fn_kwargs)
        if response is not None:
          self.active_models.add(response)
        return response
      return _impl

  def __init__(self, available_models: list[serving_model.SageMakerModel]):
    super().__init__(available_models)
    self.active_models = set(available_models)




  @classmethod
  def find_least_loaded_variant(cls, potential_variants):
    return potential_variants[0]

  @Decorators.mark_model_active
  def pick_model(self, *args, **kwargs) -> serving_model.SageMakerModel:
    log.info(f"pick_model: {kwargs}")
    for m_name in sorted(self.models.keys()):
      log.info(f"{m_name} : {self.models[m_name]._exec_latency} ({self.models[m_name].latency})")

    min_accuracy = 0.5 if "min_accuracy" not in kwargs else kwargs["min_accuracy"]
    max_latency = 100 if "max_latency" not in kwargs else kwargs["max_latency"]

    if "model_name" in kwargs:
      return self.models[kwargs["model_name"]]
    elif "min_accuracy" in kwargs:
      potential_variants = sorted(list(filter(
        (lambda m: (m.accuracy >= min_accuracy)),
        self.active_models
      )), key=(lambda m: m.latency), reverse=False)
    elif "max_latency" in kwargs:
      for m in self.active_models:
        log.info(f"{m} : {m.latency}")
      potential_variants = sorted(list(filter(
        (lambda m: (m.latency <= max_latency)),
        self.active_models
      )), key=(lambda m: m.accuracy), reverse=True)
    else:
      potential_variants = []
    log.info(f"potential_variants: {[str(m) for m in potential_variants]}")

    if len(potential_variants) > 0:
      return self.find_least_loaded_variant(potential_variants)

    # Next, search for any variant that satisfies with the lowest latency
    potential_variants = list(filter(
      (lambda m: (m.accuracy >= min_accuracy)),
      self.models.values()
    ))
    if len(potential_variants) == 0:
      raise exceptions.NoModelFound("No model found that can satisfy requirements")
    return min(potential_variants, key=(lambda m: m.latency))

  def get_models(self):
    return self.models.values()


