import depthai as dai
import cv2
import numpy as np
import argparse
from typing import Any

from utils import *
from pipeline import create_pipeline
from detection_nn_utils import to_tensor_result

parser = argparse.ArgumentParser(prog='Gerwazy')

def main():
	pipeline = create_pipeline()

	with dai.Device(pipeline) as device:

		# if verbose:
		print('Availlable streams:')
		print(f' -> Input streams: {", ".join(device.getInputQueueNames())}')
		print(f' -> Output streams: {", ".join(device.getOutputQueueNames())}')

		cam_manip_cfg_in_q: dai.DataInputQueue = device.getInputQueue('cam_manip_cfg', 1, blocking=True)
		cam_ctrl_in_q: dai.DataInputQueue = device.getInputQueue('cam_control', 1, blocking=True)
		cam_ctrl_in_q.send(camera_ctrl())

		video_q: dai.DataOutputQueue = device.getOutputQueue('video', 1, blocking=False)
		preview_q: dai.DataOutputQueue = device.getOutputQueue('preview',1, blocking=False)
		detNN_q: dai.DataOutputQueue = device.getOutputQueue('detNN_output', 1, blocking=False)
		detNN_pass_q: dai.DataOutputQueue = device.getOutputQueue('detNN_passthrough', 1, blocking=True)

		while True:
			video_frame: dai.ImgFrame = video_q.tryGet()
			preview_frame: dai.ImgFrame = preview_q.tryGet()
			passthrough_frame: dai.ImgFrame = detNN_pass_q.tryGet()
			detNN_output: dai.NNData = detNN_q.tryGet()
			detNN_pass_frame: dai.ImgFrame = detNN_pass_q.tryGet()


			if passthrough_frame is not None and detNN_output is not None:
				frame = passthrough_frame.getCvFrame()
				detNN_tensor: dict[Any, np.array] = to_tensor_result(detNN_output)
    
				scores: np.array = detNN_tensor['feature_fusion/Conv_7/Sigmoid']# confidences
				bbox_data: np.array = detNN_output['feature_fusion/mul_6'] 			# rectangular bounding boxes
				angles_data: np.array = detNN_output['feature_fusion/sub/Fused_Add_'] # angles to rotate boxes by
    
				scores = np.reshape(scores, (1, 1, 64, 64))
				bbox_data = np.reshape(bbox_data, (1, 4, 64, 64))
				angles_data = np.reshape(angles_data, (1, 1, 64, 64))

				cv2.imshow('', frame)

			if cv2.waitKey(1) == ord('q'):
				break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Gerwazy')
    main()
