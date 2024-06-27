import os
import sys
import os.path
import cv2
from src.controllers.extract_keyframe.extracting_candidate_frames import FrameExtractor
from src.controllers.extract_keyframe.clustering_with_hdbscan import ImageSelector

class KeyFrameExtractor():
    def __init__(self) -> None:
        pass

    def extract(self, video_path):
        folder_video_image_path = "data/keyframe_extract/video_image"
        folder_video_final_image_path = 'data/keyframe_extract/video_final_image'
        vd = FrameExtractor()
        
        # check if folder exists
        if not os.path.exists("data/keyframe_extract"):
            os.makedirs("data/keyframe_extract")
        if not os.path.exists(folder_video_image_path):
            os.makedirs(folder_video_image_path)
        if not os.path.exists(folder_video_final_image_path):
            os.makedirs(folder_video_final_image_path)

        imgs=vd.extract_candidate_frames(video_path)
        for counter, img in enumerate(imgs):
            vd.save_frame_to_disk(
                img,
                file_path=folder_video_image_path,
                file_name="test_" + str(counter),
                file_ext=".jpeg",
            )
        final_images = ImageSelector()
        imgs_final = final_images.select_best_frames(imgs,folder_video_image_path)
        for counter, i in enumerate(imgs_final):
            vd.save_frame_to_disk(
                i,
                file_path=folder_video_final_image_path,
                file_name="finnal_" + str(counter),
                file_ext=".jpeg",
            )
