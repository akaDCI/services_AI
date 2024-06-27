import cv2
import shutil
import os
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt
from os.path import join
import numpy as np
import torch

class YoloCrackSeg():
    def __init__(self) -> None:
        self.model = YOLO("models/yolov8x_crack_seg.pt")
        self.confidence_threshold = 0.25
        self.out_pred_dir = None
        self.out_viz_dir = None
        self.crack_predict_results = "data/crack_results/crack_predict_results"
        self.crack_viz_results = "data/crack_results/crack_viz_results"

    def infer(self, img_folder, save_results=True):
        img_dir = f"tmp/upload_files/{img_folder}"

        if save_results:
            self.out_pred_dir = f"{self.crack_predict_results}/{img_folder}"
            self.out_viz_dir = f"{self.crack_viz_results}/{img_folder}"

        if self.out_viz_dir is not None:
            for path in Path(self.crack_viz_results).glob('*'):
                shutil.rmtree(str(path))
            os.makedirs(self.out_viz_dir, exist_ok=True)

        if self.out_pred_dir is not None:
            for path in Path(self.crack_predict_results).glob('*'):
                shutil.rmtree(str(path))
            os.makedirs(self.out_pred_dir, exist_ok=True)

        paths = [path for path in Path(img_dir).glob('*.*')]
        raw_arr_imgs = []
        pred_arr_imgs = []
        for img_path in paths:
            img = Image.open(str(img_path))
            img = np.asarray(img)
            raw_arr_imgs.append(img)  # Store the raw image
            
            # Make prediction
            results = self.model.predict(source=img, conf=self.confidence_threshold, stream=True)
            img_pil = Image.fromarray(img)

            # img_pil.show()

            # Extract the segmentation mask
            for result in results:
                # get array results
                masks = result.masks.data
                boxes = result.boxes.data
                
                # extract classes
                clss = boxes[:, 5]
                # get indices of results where class is 0
                crack_indices = torch.where(clss == 0)
                # use these indices to extract the relevant masks
                crack_masks = masks[crack_indices]
                # scale for visualizing results
                crack_mask = (torch.any(crack_masks, dim=0) * 255).to(torch.uint8)
                # convert to numpy array
                crack_mask = crack_mask.cpu().numpy()
                # convert to PIL image
                crack_mask_pil = Image.fromarray(crack_mask)
                # Resize the segmentation image to match the original image size
                seg_image_pil = crack_mask_pil.resize(img_pil.size, resample=Image.NEAREST)
                seg_img_arr = np.asarray(seg_image_pil)
                
                # append to list of predicted images
                pred_arr_imgs.append(seg_img_arr)  # Store the predicted image

                if self.out_pred_dir is not None:
                    seg_image_pil.convert('RGB').save(join(self.out_pred_dir, f'yolo_{img_path.stem}.jpg'))

                if self.out_viz_dir is not None:
                    fig = plt.figure(figsize=(10, 5))
                    fig.suptitle(f'YOLOv8 Model \n img={img_path.stem} \n threshold = {self.confidence_threshold}')
                    ax = fig.add_subplot(131)
                    ax.imshow(img_pil)
                    ax = fig.add_subplot(132)
                    ax.imshow(seg_image_pil)
                    ax = fig.add_subplot(133)
                    ax.imshow(img_pil)
                    ax.imshow(seg_image_pil, alpha=0.4)
                    plt.savefig(join(self.out_viz_dir, f'yolo_{img_path.stem}.jpg'), dpi=500)
                    plt.close('all')

        return raw_arr_imgs, pred_arr_imgs