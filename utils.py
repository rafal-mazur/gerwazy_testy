import numpy as np
import depthai as dai

__all__ = ['Message', 'to_tensor_result', 'camera_ctrl']

# TODO
class Message:
    def __init__(self, content: list[str]) -> None:
        self.content = content


def to_tensor_result(packet: dai.NNData) -> dict[str: np.array]:
    return {tensor.name: np.array(packet.getLayerFp16(tensor.name)) for tensor in packet.getRaw().tensors}


# TODO: dobrać ustawienia, żeby robić najlepsze zdjęcia
def camera_ctrl() -> dai.CameraControl:
    ctrl = dai.CameraControl()

    return ctrl
