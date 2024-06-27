from src.controllers.crack_detection.yolo import YoloCrackSeg
from src.controllers.crack_detection.unet import UnetCrackSeg

yolo_seg = YoloCrackSeg()
yolo_seg.infer("98335f2c07a64271b58c4f2f2f418599")

unet_seg = UnetCrackSeg()
unet_seg.infer("98335f2c07a64271b58c4f2f2f418599")