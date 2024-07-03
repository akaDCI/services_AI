import os
from pathlib import Path
import shutil
import onnxruntime as ort
import numpy as np
from PIL import Image
import math
import matplotlib.pyplot as plt
from copy import deepcopy
import cv2 as cv


class FormerCrackSeg():
    def __init__(self, threshold=0.7):
        self.model_path = "models/seg_former.onnx"
        self.session = ort.InferenceSession(self.model_path)
        self.input_model_shape = self.session.get_inputs()[0].shape
        self.output_model_shape = self.session.get_outputs()[0].shape
        self.input_model_name = self.session.get_inputs()[0].name
        self.output_model_name = self.session.get_outputs()[0].name
        self.out_pred_dir = None
        self.out_viz_dir = None
        self.threshold = threshold*255
        self.crack_predict_results = "data/crack_results/crack_predict_results"
        self.crack_viz_results = "data/crack_results/crack_viz_results"

    def _sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    # Preprocess the image
    def _preprocess_image(self, pil_image):
        # Resize to the model's input shape
        re_image = pil_image.resize(
            (self.input_model_shape[2], self.input_model_shape[3]))
        image_data = np.asarray(re_image).astype('float32')
        image_data = image_data.transpose(2, 0, 1)  # HWC to CHW
        image_data = np.expand_dims(image_data, axis=0)  # Add batch dimension

        return image_data

    def infer(self, img_folder, save_results=True):
        img_dir = f"tmp/upload_files/{img_folder}"
        seg_results = []

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
            pil_img = Image.open(img_path).convert('RGB')
            # print image shape
            image_w, image_h = pil_img.size
            input_data = self._preprocess_image(pil_img)

            # Run inference
            ort_inputs = {self.input_model_name: input_data}
            prediction = self.session.run([self.output_model_name], ort_inputs)
            mask_seg_prediction = prediction[0][:, :, 1]
            mask_seg = cv.resize(mask_seg_prediction,
                                 (image_w, image_h), cv.INTER_AREA)

            # Convert to black and white image
            crack_mask = mask_seg * 255
            prob_map_viz_full = deepcopy(crack_mask)
            prob_map_viz_full[prob_map_viz_full < self.threshold] = 0
            pred_arr_img = (prob_map_viz_full).astype(np.uint8)

            # add to array
            arr_img = np.asarray(pil_img)
            raw_arr_imgs.append(arr_img)
            pred_arr_imgs.append(pred_arr_img)

            # convert to PIL image
            crack_mask_pil = Image.fromarray(pred_arr_img).convert("L")

            if self.out_pred_dir is not None:
                crack_mask_pil.save(os.path.join(
                    self.out_pred_dir, f'segfomer_{img_path.stem}_mask.jpg'))
                seg_results.append(os.path.join(
                    self.out_pred_dir, f'segfomer_{img_path.stem}_mask.jpg'))

            if self.out_viz_dir is not None:
                fig = plt.figure(figsize=(10, 5))
                fig.suptitle(
                    f'SegFormerCrack Model \n img={img_path.stem} \n threshold = {self.threshold/255}')
                ax = fig.add_subplot(131)
                ax.imshow(pil_img)
                ax = fig.add_subplot(132)
                ax.imshow(pred_arr_img)
                ax = fig.add_subplot(133)
                ax.imshow(pil_img)
                ax.imshow(pred_arr_img, alpha=0.4)
                # plt.show()
                plt.savefig(os.path.join(self.out_viz_dir,
                            f'segfomer_{img_path.stem}_viz.jpg'), dpi=500)
                plt.close('all')

                seg_results.append(os.path.join(
                    self.out_viz_dir, f'segfomer_{img_path.stem}_viz.jpg'))

        return seg_results, raw_arr_imgs, pred_arr_imgs
