
from __future__ import print_function
import sys
import os
from argparse import ArgumentParser, SUPPRESS
import cv2
import numpy as np
import logging as log
from time import time
from openvino.inference_engine import IENetwork, IEPlugin


def main():
#######################  Device  Initialization  ########################
#  Plugin initialization for specified device and load extensions library if specified
plugin = IEPlugin(device="MYRIAD")
#########################################################################

#########################  Load Neural Network  #########################
#  Read in Graph file (IR)
net = IENetwork.from_ir(model="graph1.xml", weights="graph1.bin")

input_blob = next(iter(net.inputs))
out_blob = next(iter(net.outputs))
#  Load network to the plugin
exec_net = plugin.load(network=net)
del net
########################################################################

#########################  Obtain Input Tensor  ########################
#  Obtain and preprocess input tensor (image)
#  Read and pre-process input image  maybe we don't need to show these details
image = cv2.imread("input_image.jpg")

#  Preprocessing is neural network dependent maybe we don't show this
n, c, h, w = net.inputs[input_blob]
image = cv2.resize(image, (w, h))
image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
image = image.reshape((n, c, h, w))
########################################################################

##########################  Start  Inference  ##########################
#  Start synchronous inference and get inference result
req_handle = exec_net.start_async(inputs={input_blob: image})
########################################################################

######################## Get Inference Result  #########################
status = req_handle.wait()
res = req_handle.outputs[out_blob


# Do something with the results... (like print top 5)
top_ind = np.argsort(res[out_blob], axis=1)[0, -5:][::-1]
for i in top_ind:
    print("%f #%d" % (res[out_blob][0, i], i))

###############################  Clean  Up  ############################
del exec_net
del plugin
########################################################################


if __name__ == '__main__':
sys.exit(main() or 0)
