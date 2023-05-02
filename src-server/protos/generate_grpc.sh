#!env bash


/Users/ssogden/miniforge3/envs/cremebrulee-env/bin/python \
  -m grpc_tools.protoc \
  -I. \
  --python_out=.. \
  --grpc_python_out=.. \
  *.proto
