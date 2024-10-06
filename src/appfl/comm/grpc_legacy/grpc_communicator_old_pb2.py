# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: grpc_communicator_old.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1bgrpc_communicator_old.proto\".\n\x06Header\x12\x11\n\tserver_id\x18\x01 \x01(\r\x12\x11\n\tclient_id\x18\x02 \x01(\r\"0\n\x0c\x44\x61taBufferV0\x12\x0c\n\x04size\x18\x01 \x01(\r\x12\x12\n\ndata_bytes\x18\x02 \x01(\x0c\"I\n\x0e\x41\x63knowledgment\x12\x17\n\x06header\x18\x01 \x01(\x0b\x32\x07.Header\x12\x1e\n\x06status\x18\x02 \x01(\x0e\x32\x0e.MessageStatus\"=\n\nJobRequest\x12\x17\n\x06header\x18\x01 \x01(\x0b\x32\x07.Header\x12\x16\n\x08job_done\x18\x03 \x01(\x0e\x32\x04.Job\"T\n\x0bJobResponse\x12\x17\n\x06header\x18\x01 \x01(\x0b\x32\x07.Header\x12\x14\n\x0cround_number\x18\x02 \x01(\r\x12\x16\n\x08job_todo\x18\x03 \x01(\x0e\x32\x04.Job\"\x8d\x01\n\x0fLearningResults\x12\x17\n\x06header\x18\x01 \x01(\x0b\x32\x07.Header\x12\x14\n\x0cround_number\x18\x02 \x01(\r\x12\x0f\n\x07penalty\x18\x03 \x01(\x02\x12\x1d\n\x06primal\x18\x04 \x03(\x0b\x32\r.TensorRecord\x12\x1b\n\x04\x64ual\x18\x05 \x03(\x0b\x32\r.TensorRecord\"L\n\rTensorRequest\x12\x17\n\x06header\x18\x01 \x01(\x0b\x32\x07.Header\x12\x0c\n\x04name\x18\x02 \x01(\t\x12\x14\n\x0cround_number\x18\x03 \x01(\r\"X\n\x0cTensorRecord\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x12\n\ndata_shape\x18\x02 \x03(\x05\x12\x12\n\ndata_bytes\x18\x03 \x01(\x0c\x12\x12\n\ndata_dtype\x18\x04 \x01(\t\"6\n\rWeightRequest\x12\x17\n\x06header\x18\x01 \x01(\x0b\x32\x07.Header\x12\x0c\n\x04size\x18\x02 \x01(\r\"9\n\x0eWeightResponse\x12\x17\n\x06header\x18\x01 \x01(\x0b\x32\x07.Header\x12\x0e\n\x06weight\x18\x02 \x01(\x02*0\n\x03Job\x12\x08\n\x04INIT\x10\x00\x12\n\n\x06WEIGHT\x10\x01\x12\t\n\x05TRAIN\x10\x02\x12\x08\n\x04QUIT\x10\x03*\"\n\rMessageStatus\x12\x06\n\x02OK\x10\x00\x12\t\n\x05\x45MPTY\x10\x01\x32\xdc\x01\n\x12GRPCCommunicatorV0\x12%\n\x06GetJob\x12\x0b.JobRequest\x1a\x0c.JobResponse\"\x00\x12\x34\n\x0fGetTensorRecord\x12\x0e.TensorRequest\x1a\r.DataBufferV0\"\x00\x30\x01\x12.\n\tGetWeight\x12\x0e.WeightRequest\x1a\x0f.WeightResponse\"\x00\x12\x39\n\x13SendLearningResults\x12\r.DataBufferV0\x1a\x0f.Acknowledgment\"\x00(\x01\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'grpc_communicator_old_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _JOB._serialized_start=780
  _JOB._serialized_end=828
  _MESSAGESTATUS._serialized_start=830
  _MESSAGESTATUS._serialized_end=864
  _HEADER._serialized_start=31
  _HEADER._serialized_end=77
  _DATABUFFERV0._serialized_start=79
  _DATABUFFERV0._serialized_end=127
  _ACKNOWLEDGMENT._serialized_start=129
  _ACKNOWLEDGMENT._serialized_end=202
  _JOBREQUEST._serialized_start=204
  _JOBREQUEST._serialized_end=265
  _JOBRESPONSE._serialized_start=267
  _JOBRESPONSE._serialized_end=351
  _LEARNINGRESULTS._serialized_start=354
  _LEARNINGRESULTS._serialized_end=495
  _TENSORREQUEST._serialized_start=497
  _TENSORREQUEST._serialized_end=573
  _TENSORRECORD._serialized_start=575
  _TENSORRECORD._serialized_end=663
  _WEIGHTREQUEST._serialized_start=665
  _WEIGHTREQUEST._serialized_end=719
  _WEIGHTRESPONSE._serialized_start=721
  _WEIGHTRESPONSE._serialized_end=778
  _GRPCCOMMUNICATORV0._serialized_start=867
  _GRPCCOMMUNICATORV0._serialized_end=1087
# @@protoc_insertion_point(module_scope)
