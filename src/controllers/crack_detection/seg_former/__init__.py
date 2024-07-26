import os
import numpy as np
from PIL import Image
import math
import cv2 as cv
import gdown
from src.utils.onnx_infer import onnx_interence_session, onnx_inference


class FormerCrackSeg():
    def __init__(self):
        self.model_path = self.__download_model()
        self.session = onnx_interence_session(self.model_path)
        self.input_model_shape = self.session.get_inputs()[0].shape
        self.output_model_shape = self.session.get_outputs()[0].shape

    def __download_model(self):
        seg_former_model_path = os.path.join(
            os.getcwd(), "models", "seg_former.onnx")
        if not os.path.exists(seg_former_model_path):
            gdown.download(id="1K6pE3fexH25ek4OrrUbvIvqpg7YM1AXv",
                           output=seg_former_model_path)
        return seg_former_model_path

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

    def infer(self, images: list[Image.Image], threshold: float):
        crack_results = []
        threshold = threshold * 255

        for image in images:
            # print image shape
            image_w, image_h = image.size
            input_data = self._preprocess_image(image)

            # Run inference
            prediction = onnx_inference(self.session, input_data)
            mask_seg_prediction = prediction[:, :, 1]
            mask_seg = cv.resize(mask_seg_prediction,
                                 (image_w, image_h), interpolation=cv.INTER_AREA)

            # Convert to black and white image
            crack_mask = mask_seg * 255
            crack_mask[crack_mask < threshold] = 0
            pred_arr_img = crack_mask.astype(np.uint8)

            # Convert to PIL image
            crack_mask_result = Image.fromarray(pred_arr_img).convert("L")
            crack_results.append(crack_mask_result)

            # Save result chart
            # _title = f'SegFormerCrack Model; threshold = {threshold/255}'
            # compile_result_chart(_title, image, crack_mask_result)

        return crack_results
