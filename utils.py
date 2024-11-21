import numpy as np
import depthai as dai

__all__ = ['Message', 'camera_ctrl']

# TODO
class Message:
    def __init__(self, content: list[str]) -> None:
        self.content = content


# TODO: dobrać ustawienia, żeby robić najlepsze zdjęcia
def camera_ctrl() -> dai.CameraControl:
    ctrl = dai.CameraControl()

    return ctrl
