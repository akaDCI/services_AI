from src.controllers.crack_detection.yolo import YoloCrackSeg
from src.controllers.crack_detection.unet import UnetCrackSeg

yolo_seg = YoloCrackSeg()
yolo_seg.infer("binh_nut")

# unet_seg = UnetCrackSeg()
# unet_seg.infer("binh_nut")