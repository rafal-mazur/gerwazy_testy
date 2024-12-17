import depthai as dai
import cv2
import numpy as np
import time
import utils.communication as comm


import decoding.east256x256 as east
import decoding.text_recognition_0012 as tr12
from utils import *
from utils.geometry import RRect
from utils.pipeline import create_pipeline
import utils.Logger as Logger




def raw_read(args):
logger = Logger.Logger()
    logger('Creating pipeline...')
    pipeline: dai.Pipeline = create_pipeline()
    logger('Pipeline created!\n')

    
    with dai.Device(pipeline) as device:
        logger('USB speed:', device.getUsbSpeed().name)

        logger(f'\nAvaillable input queues: {device.getInputQueueNames()}')
        logger(f'Availlable output queues: {device.getOutputQueueNames()}\n')
        logger('Creating queues...')

        q_cam_ctrl: dai.DataInputQueue  = device.getInputQueue('cam_ctrl', 1, blocking=False)
        q_manip_img: dai.DataInputQueue = device.getInputQueue('manip_img', 4, blocking=False)
        q_manip_cfg: dai.DataInputQueue = device.getInputQueue('manip_cfg', 4, blocking=False)

        q_detnn_out: dai.DataOutputQueue  = device.getOutputQueue('detnn_out', 1, blocking=False)
        q_detnn_pass: dai.DataOutputQueue = device.getOutputQueue('detnn_pass', 1, blocking=False)
        q_manip_out: dai.DataOutputQueue  = device.getOutputQueue('manip_out', 1, blocking=False)
        q_recnn_out: dai.DataOutputQueue  = device.getOutputQueue('recnn_out', 1, blocking=False)

        logger('Queues created')

        ctrl: dai.CameraControl = dai.CameraControl()
        ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.AUTO)
        ctrl.setAutoFocusTrigger()
        q_cam_ctrl.send(ctrl)
        del ctrl

        logger('\nStarting main loop\n')

        while True:
            time.sleep(0.01)
            detnn_output: dai.NNData = q_detnn_out.get()
            detnn_pass: dai.ImgFrame = q_detnn_pass.get().getCvFrame()

            # decode detection
            for idx, (rect, _) in enumerate(east.decode(detnn_output)):
                

                cfg: dai.ImageManipConfig = dai.ImageManipConfig()
                cfg.setCropRotatedRect(rect.get_depthai_RotatedRect(), False)
                cfg.setResize(120, 32)

                if idx == 0:
                    w, h, _ = detnn_pass.shape
                    imgFrame = dai.ImgFrame()
                    imgFrame.setData(detnn_pass.transpose(2, 0, 1).flatten())
                    imgFrame.setType(dai.ImgFrame.Type.BGR888p)
                    imgFrame.setWidth(w)
                    imgFrame.setHeight(h)
                    q_manip_img.send(imgFrame)
                else:
                    cfg.setReusePreviousImage(True)
                q_manip_cfg.send(cfg)

            
            while True:
                recnn_out: dai.NNData|None = q_recnn_out.tryGet()

                if recnn_out is None:
                    break
                
                text = tr12.decode(recnn_out)
                print(text)
                # TODO: send to another device



            if cv2.waitKey(1) == ord('q'):
                break


