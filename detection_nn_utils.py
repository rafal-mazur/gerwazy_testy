import depthai as dai
import numpy as np

def to_CWH(frame: np.ndarray):
	return frame.transpose(2, 0, 1)
