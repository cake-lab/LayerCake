#!env python

import datetime
import io
import logging
import time

import boto3
import sagemaker
from PIL import Image

import serving_model

logging.basicConfig()
log = logging.getLogger("layercake.monitoring")
#log = logging.getLogger()
log.setLevel(logging.INFO)
logging.getLogger("botocore").setLevel(logging.WARNING)


class DeploymentMonitor(object):
  def __init__(self, *args, **kwargs):
    self.sagemaker_client = boto3.client("sagemaker")
    self.metrics_client = boto3.client("cloudwatch")
    self.endpoints = [] if "endpoints" not in kwargs else kwargs["endpoints"]

  def check_endpoint(self, endpoint_name):
    endpoint_description = self.sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
    endpoint_config_name = endpoint_description["EndpointConfigName"]
    max_serverless_concurrency = 0
    for variant_description in endpoint_description["ProductionVariants"]:
      if "CurrentServerlessConfig" in variant_description:
        max_serverless_concurrency += variant_description["CurrentServerlessConfig"]["MaxConcurrency"]
    log.info(f"{endpoint_name} ({max_serverless_concurrency}) : {endpoint_config_name}")


  def get_all_endpoints(self):
    active_endpoints = []
    for page in self.sagemaker_client.get_paginator("list_endpoints").paginate(**{"StatusEquals" : "InService"}):
      active_endpoints.extend([e["EndpointName"] for e in page["Endpoints"]])
    return active_endpoints

  def get_num_invocations(self, endpoint_name):

    def generate_query(metric_name : str, period : int, stat: str):
      return {
        "Id" : f"{metric_name.lower()}",
        "MetricStat" : {
          "Metric" : {
            "Namespace" : "AWS/SageMaker",
            "MetricName" : f"{metric_name}",
            "Dimensions" : [
              {
                "Name": "EndpointName",
                "Value": f"{endpoint_name}",
              },
              {
                "Name": "VariantName",
                "Value": "AllTraffic",
              },
            ],
          },
          "Period" : period,
          "Stat": f"{stat}",
        }
      }

    stats = self.metrics_client.get_metric_data(
      MetricDataQueries=[
        generate_query(*p)
        for p in [
          ("Invocations", 1, "Sum"),
          ("ModelLatency", 1, "Average"),
          ("ModelSetupTime", 1, "Average"),
          ("OverheadLatency", 1, "Average"),
          ("InvocationsPerInstance", 1, "Sum"),
        ]
      ],
      StartTime = datetime.datetime.utcnow() - datetime.timedelta(seconds = 60*60),
      EndTime = datetime.datetime.utcnow(),
    )
    for stat in stats["MetricDataResults"]:
      print(f"{stat['Id']} : {[v/1000000 for v in stat['Values']]}")




if __name__ == "__main__":
  endpoint_name = "tensorflow-inference-2022-05-23-01-56-52-262"
  predictor = sagemaker.Predictor(endpoint_name=endpoint_name, serializer=serving_model.ImageB64Serializer())
  data = io.BytesIO()
  Image.open("mug.jpg").resize((224,224)).save(data, format="JPEG")
  predictor.predict(data.getvalue())

  monitor = DeploymentMonitor()
  time.sleep(30)
  monitor.get_num_invocations(endpoint_name)