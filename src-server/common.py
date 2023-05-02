#!env python

import enum
import logging

logging.basicConfig()
log = logging.getLogger("common")
log.setLevel(logging.INFO)

# Built-in imports
import time
import json
import os

# Pip'd imports
import wrapt
import numpy as np

# My imports
## There should be none


#############
## Globals ##
#############
TIMEOUT_IN_SECONDS = None  # 600
POLL_SLEEP_MS = 100
MAX_TIME_UNTIL_LIVE = 60 * 1000
REPORT_INTERVAL = 1
PLACEMENT_PERIOD = 2

DUMMY_EXEC_LATENCY = 0.000
DUMMY_LOAD_LATENCY = 0.000
DUMMY_UNLOAD_LATENCY = 0

KEEP_ALIVE_IN_SECONDS = 2
PLACEMENT_POLL_INTERVAL = 0.1

MODEL_INFO_FRESHNESS = 5.
MAX_MEMORY_IN_GB = 2.
RNG_SEED = 0
NUM_MEASUREMENTS = 100
NUM_REQUESTS_PER_MEASUREMENT = 10
MODEL_REPO = os.path.abspath("../models")
MODELS_FILE = os.path.abspath("./model_stats.json")

NUM_EXECUTION_THREADS = 1
SHUTDOWN_GRACE = 3


#############
#############

class Application(str, enum.Enum):
  IMAGE = "image",
  TEXT = "text"


data = None


def get_models_present():
  model_directory = MODEL_REPO
  models_present = list(
    filter(
      (lambda d: os.path.isdir(os.path.join(model_directory, d))),
      os.listdir(model_directory)
    )
  )
  return models_present


def load_models_info(stats_file_path):
  def safe_convert(model_info):
    for (field_name, field_data) in model_info.items():
      try:
        model_info[field_name] = float(field_data)
      except ArithmeticError:
        model_info[field_name] = field_data
    return model_info

  model_dict = json.load(open(stats_file_path))
  for (model_name, model_info) in model_dict.items():
    model_dict[model_name] = safe_convert(model_info)
  return model_dict


@wrapt.decorator
def timing(wrapped, instance, args, kwargs):
  def wrapper(*args, **kw):
    ts = time.time()
    result = wrapped(*args, **kw)
    te = time.time()
    log.info(f"TIMING: {wrapped.__name__}({[str(i)[:100] for i in args], kw}): {te - ts}s")
    return result

  return wrapper(*args, **kwargs)


def logInfo(event_type):
  @wrapt.decorator
  def wrapper(wrapped, instance, args, kwargs):
    if instance.record:
      start_mem = instance.server_container.stats(stream=False)["memory_stats"]["usage"]
    t_start = time.time()
    result = wrapped(*args, **kwargs)
    t_end = time.time()
    if instance.record:
      end_mem = instance.server_container.stats(stream=False)["memory_stats"]["usage"]

    if instance.record:
      instance.write_to_log(instance.get_name, event_type, "MEMORY", start_mem, {"state": "prior"})
      instance.write_to_log(instance.get_name, event_type, "MEMORY", end_mem, {"state": "post"})
      instance.write_to_log(instance.get_name, event_type, "MEMORY", end_mem - start_mem, {"state": "delta"})
      instance.write_to_log(instance.get_name, event_type, "TIME", (t_end - t_start))

    return result

  return wrapper

