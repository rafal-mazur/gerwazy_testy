import depthai as dai
import cv2
import numpy as np
import argparse

from geometry import RotatedRect, Point
from east import decode_east, to_tensor_result
from utils import *
from pipeline import create_pipeline


parser = argparse.ArgumentParser(prog='Gerwazy')

def main():
    pipeline = create_pipeline()
    print('QAAAAA')
    with dai.Device(pipeline) as device:

        # if verbose:
        print('Availlable streams:')
        print(f' -> Input streams: {", ".join(device.getInputQueueNames())}')
        print(f' -> Output streams: {", ".join(device.getOutputQueueNames())}')

        cam_manip_cfg_in_q: dai.DataInputQueue = device.getInputQueue('cam_manip_cfg', 1, blocking=True)
        cam_ctrl_in_q: dai.DataInputQueue = device.getInputQueue('cam_control', 1, blocking=True)
        cam_ctrl_in_q.send(camera_ctrl())
        imgmanip_img_in_q :dai.DataInputQueue = device.getInputQueue('imgmanip_img')
        imgmanip_cfg_in_q :dai.DataInputQueue = device.getInputQueue('imgmanip_cfg')

        video_q: dai.DataOutputQueue = device.getOutputQueue('video', 1, blocking=False)
        preview_q: dai.DataOutputQueue = device.getOutputQueue('preview',1, blocking=False)
        detNN_q: dai.DataOutputQueue = device.getOutputQueue('detNN_output', 1, blocking=False)
        detNN_pass_q: dai.DataOutputQueue = device.getOutputQueue('detNN_passthrough', 1, blocking=True)
        imgmanip_debug_q: dai.DataOutputQueue = device.getOutputQueue('imgmanip_debug')

        while True:
            video_frame: dai.ImgFrame = video_q.tryGet()
            preview_frame: dai.ImgFrame = preview_q.tryGet()
            passthrough_frame: dai.ImgFrame = detNN_pass_q.tryGet()
            detNN_output: dai.NNData = detNN_q.tryGet()
            detNN_pass_frame: dai.ImgFrame = detNN_pass_q.tryGet()

            if passthrough_frame is not None and detNN_output is not None:
                frame = passthrough_frame.getCvFrame().astype('uint8')
                
                for rect, score in decode_east(detNN_output):
                    frame = cv2.polylines(frame, [rect.get_rotated_points()], True, (255,0,0), 1)
                    
                # cv2.imshow('frame', frame)

            # if passthrough_frame is not None and detNN_output is not None:
            #     frame = passthrough_frame.getCvFrame()
            #     sleep(0.1)
            #     for rect, score in decode_east(detNN_output):
            #         # cfg = dai.ImageManipConfig()
            #         # cfg.setCropRotatedRect(rect.get_depthai_RotatedRect(), False)
            #         # cfg.setResize(120, 32)    
            #         cv2.polylines(frame, rect.get_rotated_points(),isClosed=True, color=(255,0,0), thickness=1, lineType=cv2.LINE_8)
            #     cv2.imshow('', frame)
            
            if cv2.waitKey(1) == ord('q'):
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Gerwazy')
    main()
