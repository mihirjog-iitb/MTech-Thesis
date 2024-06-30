#!/bin/bash
/usr/src/tensorrt/bin/trtexec --onnx=DenseNet16.onnx --saveEngine=DenseNet16_engine.trt
/usr/src/tensorrt/bin/trtexec --onnx=DenseNet64.onnx --saveEngine=DenseNet64_engine.trt
/usr/src/tensorrt/bin/trtexec --onnx=mobilenet16.onnx --saveEngine=mobilenet16_engine.trt
/usr/src/tensorrt/bin/trtexec --onnx=mobilenet64.onnx --saveEngine=mobilenet64_engine.trt
/usr/src/tensorrt/bin/trtexec --onnx=resNet16.onnx --saveEngine=resNet16_engine.trt
/usr/src/tensorrt/bin/trtexec --onnx=resNet64.onnx --saveEngine=resNet64_engine.trt

