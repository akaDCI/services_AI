from src.controllers.crack_detection.yolo import YoloCrackSeg
from src.controllers.crack_detection.unet import UnetCrackSeg
from src.controllers.extract_keyframe import KeyFrameExtractor
# yolo_seg = YoloCrackSeg()
# yolo_seg.infer("binh_nut")

# unet_seg = UnetCrackSeg()
# unet_seg.infer("binh_nut")

keyframe_extractor = KeyFrameExtractor()
keyframe_extractor.extract("tmp/upload_files/cai_loa.mp4")