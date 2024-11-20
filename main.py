import depthai as dai
import cv2
import numpy as np
import argparse

from utils import *
from pipeline import create_pipeline

parser = argparse.ArgumentParser(prog='Gerwazy')

def main():
	pipeline = create_pipeline()

	with dai.Device(pipeline) as device:

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
			video_frame = video_q.tryGet()
			preview_frame = preview_q.tryGet()
			passthrough_frame= detNN_pass_q.tryGet()
			detNN_output = detNN_q.tryGet()
			detNN_pass_frame = detNN_pass_q.tryGet()


			if passthrough_frame is not None and detNN_output is not None:
				print(passthrough_frame.getSequenceNum() - detNN_output.getSequenceNum())
				frame = passthrough_frame.getCvFrame().copy()
				scores, geom1, geom2 = to_tensor_result(detNN_output).values()
				scores = np.reshape(scores, (1, 1, 64, 64))
				geom1 = np.reshape(geom1, (1, 4, 64, 64))
				geom2 = np.reshape(geom2, (1, 1, 64, 64))
				print(scores)
				print(geom1)
				print(geom2)

				cv2.imshow('', frame)

			if cv2.waitKey(1) == ord('q'):
				break


if __name__ == '__main__':
	main()
