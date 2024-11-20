import depthai as dai
import numpy as np

def to_CWH(frame: np.ndarray):
	return frame.transpose(2, 0, 1)

def to_tensor_result(packet: dai.NNData):
    return {tensor.name: np.array(packet.getLayerFp16(tensor.name)) for tensor in packet.getRaw().tensors}