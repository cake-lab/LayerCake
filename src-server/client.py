#!env python

# Built-in imports
import logging
import base64
import json
import sys
import time
import random
import threading

# pip'd imports
import grpc

# my imports
import inference_service_pb2 as inference_service_pb2
import inference_service_pb2_grpc as inference_service_pb2_grpc
import common


logging.basicConfig()
log = logging.getLogger("layercake")
log.setLevel(logging.DEBUG)

NUM_SAMPLES = 10
RANDOM_DATA = base64.b64encode(random.randbytes(31000))

class LayercakeClient:
  def __init__(self):
    self.channel = grpc.insecure_channel('3.235.51.141:50051')
    #self.channel = grpc.insecure_channel('localhost:50051')
    self.stub = inference_service_pb2_grpc.InferenceStub(self.channel)
    self.total_time = 0
    self.num_requests = 0

  def infer1(self, model_name="efficientnetb0"):
    request = inference_service_pb2.Message1(
      application=inference_service_pb2.Application.CLASSIFICATION,
      model_name=model_name,
      data=RANDOM_DATA
    )
    time_start = time.time()
    feature_future = self.stub.Infer1.future(request)
    response = feature_future.result()
    time_end = time.time()
    response_dict = json.loads(response.response[0])
    return (time_end - time_start), response_dict["model"], response_dict

  def infer2_accuracy(self, accuracy_target=0.5):
    request = inference_service_pb2.Message2(
      application=inference_service_pb2.Application.CLASSIFICATION,
      slo_type=inference_service_pb2.SLOType.ACCURACY,
      slo_value=accuracy_target,
      data=RANDOM_DATA
    )
    time_start = time.time()
    feature_future = self.stub.Infer2.future(request)
    response = feature_future.result()
    time_end = time.time()
    response_dict = json.loads(response.response[0])
    return (time_end - time_start), response_dict["model"], response_dict

  def infer2_latency(self, latency_target=1.0):
    request = inference_service_pb2.Message2(
      slo_type=inference_service_pb2.SLOType.LATENCY,
      application=inference_service_pb2.Application.CLASSIFICATION,
      slo_value=latency_target,
      data=RANDOM_DATA
    )
    time_start = time.time()
    feature_future = self.stub.Infer2.future(request)
    response = feature_future.result()
    time_end = time.time()
    response_dict = json.loads(response.response[0])
    return (time_end - time_start), response_dict["model"], response_dict

  def infer3_latency(self, accuracy_target=0.5, latency_target=1.0):
    request = inference_service_pb2.Message3(
      application=inference_service_pb2.Application.CLASSIFICATION,
      slo_type=inference_service_pb2.SLOType.LATENCY,
      slo_value=latency_target,
      data=RANDOM_DATA
    )
    time_start = time.time()
    feature_future = self.stub.Infer3.future(request)
    response = feature_future.result()
    time_end = time.time()
    response_dict = json.loads(response.response[0])
    return f"{(time_end - time_start):0.3f}s", response_dict["model"], response_dict


  def get_average_latency(self):
    return self.total_time / self.num_requests

def main():
  client = LayercakeClient()
  #client.run_inference("efficientnetb0")

  print("Testing infer1")
  for model_name in [
    "efficientnetb0",
    "efficientnetb1",
    "efficientnetb2",
    "efficientnetb3",
    "efficientnetb4",
    "efficientnetb5",
    "efficientnetb6",
    "efficientnetb7",
  ]:
    ts = time.time()
    for i in range(100):
      client.infer1(model_name)
      sys.stdout.write('.')
      sys.stdout.flush()
    te = time.time()
    sys.stdout.write('\n')
    print(client.infer1(model_name))
    print(f"{(te-ts)/100:0.3f}s")

  return

if __name__ == "__main__":
  main()