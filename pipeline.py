import depthai as dai

import settings

__all__ = ['create_pipeline']


def create_pipeline() -> dai.Pipeline:
    pipeline = dai.Pipeline()

    # ------------------------------------------------------------------------
    # Creating nodes
    # ------------------------------------------------------------------------

    cam = pipeline.create(dai.node.ColorCamera)
    cam_control = pipeline.create(dai.node.XLinkIn)
    cam_manip_config = pipeline.create(dai.node.XLinkIn)
    cam_xout_video = pipeline.create(dai.node.XLinkOut)
    cam_xout_preview = pipeline.create(dai.node.XLinkOut)


    detection_nn = pipeline.create(dai.node.NeuralNetwork)
    detection_nn_output_xout = pipeline.create(dai.node.XLinkOut)
    detection_nn_passthrough_xout = pipeline.create(dai.node.XLinkOut)

    # ------------------------------------------------------------------------
    # TODO: recognition part
    # ------------------------------------------------------------------------
    


    # ------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------

    cam.setPreviewSize(settings.Device.PREVIEW_SIZE)
    cam.setVideoSize(settings.Device.VIDEO_SIZE)
    cam.setResolution(settings.Device.SENSOR_RESOLUTION)
    cam.setInterleaved(settings.Device.INTERLEAVED)
    cam.setBoardSocket(settings.Device.BOARD_SOCKET)
    cam.setFps(settings.Device.FPS)

    cam_xout_preview.setStreamName('preview')
    cam_xout_video.setStreamName('video')
    cam_control.setStreamName('cam_control')
    cam_manip_config.setStreamName('cam_manip_cfg')

    detection_nn.setBlobPath(settings.BlobPaths.DETECTION_NETWORK)
    detection_nn_output_xout.setStreamName('detNN_output')
    detection_nn_passthrough_xout.setStreamName('detNN_passthrough')

	# ...

    # ------------------------------------------------------------------------
    # Linking
    # ------------------------------------------------------------------------
    cam_control.out.link(cam.inputControl)
    cam.video.link(cam_xout_video.input)
    cam.preview.link(cam_xout_preview.input)
    cam.preview.link(detection_nn.input)

    detection_nn.passthrough.link(detection_nn_passthrough_xout.input)
    detection_nn.out.link(detection_nn_output_xout.input)
    
	# ...

    return pipeline
