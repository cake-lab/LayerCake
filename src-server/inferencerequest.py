#!env python


## Built-in imports
import itertools
import threading
import time
import json
from functools import total_ordering
import asyncio.events

## pip'd imports
# (none)

## my imports
import common


@total_ordering
class InferenceRequest(object):
  _ids = itertools.count(1)

  def __init__(self, model_name, data, id_num=None, allow_timeout=True):
    if id_num is None:
      self.id = next(self._ids)
    else:
      self.id = id_num
    self.model_name = model_name
    self.model = None
    self.data = data
    
    self.complete = threading.Event()
    self.times = {
      "entry_time" : time.time(),
      "assignment_time" : 0.,
      "execution_start_time" : 0.,
      "execution_end_time" : 0.,
    }
    self.model_miss = False
    self.response = None

  def __repr__(self):
    return f"<{self.__class__.__name__}: {self.id}, {self.model_name}, \"{self.response}\">"

  def __str__(self):
    return repr(self)

  def __lt__(self, other):
    return self.times["entry_time"] < other.times["entry_time"]

  def __eq__(self, other):
    return self.times["entry_time"] == other.times["entry_time"]

  def set_model(self, model):
    self.model = model

  def mark_assigned(self, miss=False):
    self.times["assignment_time"] = time.time()
    if miss:
      self.mark_model_miss()

  def mark_execution_start(self, miss=False):
    if miss:
      self.mark_model_miss()
    if self.is_started():
      return False
    self.times["execution_start_time"] = time.time()
    return True

  def mark_execution_end(self):
    self.times["execution_end_time"] = time.time()
    self.complete.set()
  
  def mark_model_miss(self):
    self.model_miss = True

  def is_started(self):
    return (self.times["execution_start_time"] != 0.)

  def getResponse(self):
    response_dict = {
      "model" : self.model_name,
      "response" : self.response,
      "placement_delay" : self.times["assignment_time"] - self.times["entry_time"],
      "queue_delay" : self.times["execution_start_time"] - self.times["assignment_time"],
      "execution_delay" : self.times["execution_end_time"] - self.times["execution_start_time"],
      "overall_latency" : self.times["execution_end_time"] - self.times["entry_time"],
      "model_miss" : self.model_miss,
    }
    return json.dumps(response_dict)
