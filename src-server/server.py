#!env python

# Built-in imports
import io
import json
import threading
from concurrent import futures
import time
import base64
import os
import logging

# pip'd imports
import PIL.Image
import grpc

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# my imports
import inference_service_pb2 as inference_service_pb2
import inference_service_pb2_grpc as inference_service_pb2_grpc
import inferencerequest
import common
import model_selection
import serving_model
import deployment_monitor

logging.basicConfig()
log = logging.getLogger("layercake")
log.setLevel(logging.INFO)
logging.getLogger("botocore").setLevel(logging.WARNING)



class InferenceServicer(inference_service_pb2_grpc.InferenceServicer):
  def __init__(self, *args, **kwargs):
    super().__init__()

    endpoints = serving_model.SageMakerModel.setup_available_models()
    self.model_selection = model_selection.INFaaSModelPicker(endpoints.values())
    self.monitor = deployment_monitor.DeploymentMonitor()
    #for endpoint in self.monitor.get_all_endpoints():
    #  self.monitor.check_endpoint(endpoint)
    for model_name in sorted(endpoints.keys()):
      log.info(f"Cycling {model_name} ({endpoints[model_name]})")
      endpoints[model_name].cycle_model(num_executions=100)
      log.info(f"Deployed {model_name} as {endpoints[model_name].endpoint.endpoint_name} ")

    for model_name in sorted(endpoints.keys()):
      log.info(f"{model_name} ({endpoints[model_name].exec_latency})")

  def Infer(self, request, context):
    log.debug(f"Got request: {request} for {request.model_name}")
    request_type = request.type
    model_name_list = request.model_name
    input_data = [base64.b64decode(d) for d in request.data]
    if request.flags is not None and request.flags != "":
      log.debug(f"flags: {request.flags}")
      flags = json.loads(base64.b64decode(request.flags))
    else:
      flags = {}

    model_to_use = self.model_selection.pick_model()
    response = model_to_use.infer(do_resize=True)

    response_dict = {
      "time" : time.time(), # In case we want to synchronize time in the most basic way
      "estimated_queue_delay" : 0.0, # Estimated time for a request to wait if it were in the system now
      "estimated_exec_delay" : 0.0, # Estimated execution latency
      #"models_loaded" : list(self.server.available_models.keys()), # List of models that would be loaded if the request arrived right now
      "response": response
    }
    response = inference_service_pb2.InferenceResponse(response=[json.dumps(response_dict)])
    return response

  def Infer1(self, request, context):
    ts = time.time()
    print(f"Infer1")
    application = request.application
    model_name = request.model_name

    model_to_use = self.model_selection.pick_model(model_name=model_name.lower())

    data = self.process_data(model_to_use, request)
    response = model_to_use.infer(data)
    return inference_service_pb2.InferenceResponse(metadata=inference_service_pb2.ServerMetadata(processing_latency=(time.time()-ts)), response=[json.dumps({"response" : response, "model" : model_to_use.name, "infer_cold": str(model_to_use.measurements[serving_model.ActionState.INFER_COLD]), "infer": str(model_to_use.measurements[serving_model.ActionState.INFER])})])

  def Infer2(self, request, context):
    ts = time.time()
    print(f"Infer2:")
    application = request.application
    slo_type = request.slo_type
    slo_value = request.slo_value

    if slo_type == inference_service_pb2.SLOType.ACCURACY:
      model_to_use = self.model_selection.pick_model(min_accuracy=slo_value)
    else:
      model_to_use = self.model_selection.pick_model(max_latency=slo_value)

    data = self.process_data(model_to_use, request)
    response = model_to_use.infer(data)
    return inference_service_pb2.InferenceResponse(
      metadata=inference_service_pb2.ServerMetadata(processing_latency=(time.time()-ts)),
      response=[
        json.dumps(
          {
            "response" : response,
            "model" : model_to_use.name,
            "infer_cold": str(model_to_use.measurements[serving_model.ActionState.INFER_COLD]),
            "infer": str(model_to_use.measurements[serving_model.ActionState.INFER]),
            "accuracy": model_to_use.accuracy
          }
        )
      ],
      endpoint=inference_service_pb2.EndpointInformation(
        model_name=model_to_use.name,
        endpoint_name=model_to_use.endpoint.endpoint_name,
        accuracy=model_to_use.accuracy,
        latency=model_to_use.exec_latency,
        dimensions=model_to_use.dimensions[0],
        application=(
          inference_service_pb2.Application.IMAGE
          if model_to_use.application == common.Application.IMAGE
          else inference_service_pb2.Application.TEXT
        )
      )
    )

  def Infer3(self, request, context):
    ts = time.time()
    print("Infer3")
    application = request.application
    min_accuracy = request.accuracy_slo
    max_latency = request.latency_slo

    if request.application == inference_service_pb2.IMAGE:
      application = common.Application.IMAGE
    else:
      application = common.Application.TEXT

    appropriate_models = list(filter((lambda m: m.accuracy >= min_accuracy and m.latency <= max_latency and m.application == application), self.model_selection.get_models()))

    return inference_service_pb2.Endpoints(
      metadata=inference_service_pb2.ServerMetadata(processing_latency=(time.time()-ts)),
      endpoints=[
        inference_service_pb2.EndpointInformation(
          model_name=model.name,
          endpoint_name=model.endpoint.endpoint_name,
          accuracy=model.accuracy,
          latency=model.exec_latency,
          dimensions=model.dimensions[0],
          application=(
            inference_service_pb2.Application.IMAGE
            if model.application == common.Application.IMAGE
            else inference_service_pb2.Application.TEXT
          )
        )
        for model in appropriate_models
      ]
    )

  def BandwidthMeasurement(self, request, context):
    """Just return the data so we can get a measurement on the other end"""
    log.info(f"BandwidthMeasurement recieved ({len(request.data)}bytes)")
    return inference_service_pb2.BandwidthMeasurementMessage(data=request.data)

  @staticmethod
  def process_data(model_selected: serving_model.SageMakerModel, request):
    """Reads in the data and decrypts from b64 to an image (hopefully correctly but I'll find out I suppose!)"""
    #img_byte_arr = io.BytesIO()
    #base64.decode(request.data, img_byte_arr)
    img_byte_arr = io.BytesIO(base64.urlsafe_b64decode(request.data))
    img = PIL.Image.open(img_byte_arr)
    return img

def serve():
  try:
    inference_servicer = InferenceServicer()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    inference_service_pb2_grpc.add_InferenceServicer_to_server(
      inference_servicer, server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("server started")
    server.wait_for_termination()
    log.info("Shutting down")
  finally:
    serving_model.SageMakerModelEndpoint.document_active_endpoints()
  exit(0)


def main():
  serve()


if __name__ == "__main__":
  main()
