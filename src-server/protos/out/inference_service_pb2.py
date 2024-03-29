# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: inference_service.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='inference_service.proto',
  package='inference_service',
  syntax='proto2',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x17inference_service.proto\x12\x11inference_service\"q\n\x10InferenceRequest\x12\x12\n\nmodel_name\x18\x01 \x03(\t\x12,\n\x04type\x18\x02 \x02(\x0e\x32\x1e.inference_service.RequestType\x12\x0c\n\x04\x64\x61ta\x18\x03 \x03(\t\x12\r\n\x05\x66lags\x18\x04 \x01(\t\"%\n\x11InferenceResponse\x12\x10\n\x08response\x18\x01 \x03(\t*\'\n\x0bRequestType\x12\x08\n\x04INFO\x10\x00\x12\x0e\n\nSUBMISSION\x10\x01\x32\x61\n\tInference\x12T\n\x05Infer\x12#.inference_service.InferenceRequest\x1a$.inference_service.InferenceResponse\"\x00'
)

_REQUESTTYPE = _descriptor.EnumDescriptor(
  name='RequestType',
  full_name='inference_service.RequestType',
  filename=None,
  file=DESCRIPTOR,
  create_key=_descriptor._internal_create_key,
  values=[
    _descriptor.EnumValueDescriptor(
      name='INFO', index=0, number=0,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='SUBMISSION', index=1, number=1,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=200,
  serialized_end=239,
)
_sym_db.RegisterEnumDescriptor(_REQUESTTYPE)

RequestType = enum_type_wrapper.EnumTypeWrapper(_REQUESTTYPE)
INFO = 0
SUBMISSION = 1



_INFERENCEREQUEST = _descriptor.Descriptor(
  name='InferenceRequest',
  full_name='inference_service.InferenceRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='model_name', full_name='inference_service.InferenceRequest.model_name', index=0,
      number=1, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='type', full_name='inference_service.InferenceRequest.type', index=1,
      number=2, type=14, cpp_type=8, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='data', full_name='inference_service.InferenceRequest.data', index=2,
      number=3, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='flags', full_name='inference_service.InferenceRequest.flags', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=46,
  serialized_end=159,
)


_INFERENCERESPONSE = _descriptor.Descriptor(
  name='InferenceResponse',
  full_name='inference_service.InferenceResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='response', full_name='inference_service.InferenceResponse.response', index=0,
      number=1, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=161,
  serialized_end=198,
)

_INFERENCEREQUEST.fields_by_name['type'].enum_type = _REQUESTTYPE
DESCRIPTOR.message_types_by_name['InferenceRequest'] = _INFERENCEREQUEST
DESCRIPTOR.message_types_by_name['InferenceResponse'] = _INFERENCERESPONSE
DESCRIPTOR.enum_types_by_name['RequestType'] = _REQUESTTYPE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

InferenceRequest = _reflection.GeneratedProtocolMessageType('InferenceRequest', (_message.Message,), {
  'DESCRIPTOR' : _INFERENCEREQUEST,
  '__module__' : 'inference_service_pb2'
  # @@protoc_insertion_point(class_scope:inference_service.InferenceRequest)
  })
_sym_db.RegisterMessage(InferenceRequest)

InferenceResponse = _reflection.GeneratedProtocolMessageType('InferenceResponse', (_message.Message,), {
  'DESCRIPTOR' : _INFERENCERESPONSE,
  '__module__' : 'inference_service_pb2'
  # @@protoc_insertion_point(class_scope:inference_service.InferenceResponse)
  })
_sym_db.RegisterMessage(InferenceResponse)



_INFERENCE = _descriptor.ServiceDescriptor(
  name='Inference',
  full_name='inference_service.Inference',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=241,
  serialized_end=338,
  methods=[
  _descriptor.MethodDescriptor(
    name='Infer',
    full_name='inference_service.Inference.Infer',
    index=0,
    containing_service=None,
    input_type=_INFERENCEREQUEST,
    output_type=_INFERENCERESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_INFERENCE)

DESCRIPTOR.services_by_name['Inference'] = _INFERENCE

# @@protoc_insertion_point(module_scope)
