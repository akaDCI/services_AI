import os
import torch
import numpy as np
from pathlib import Path
import cv2 as cv
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from src.controllers.crack_detection.unet.unet_transfer import UNet16, input_size
import matplotlib.pyplot as plt
from os.path import join
from PIL import Image
import gc
import shutil
from src.controllers.crack_detection.unet.utils import load_unet_vgg16, load_unet_resnet_101, load_unet_resnet_34

class UnetCrackSeg():
    def __init__(self):
        self.channel_means = [0.485, 0.456, 0.406]
        self.channel_stds  = [0.229, 0.224, 0.225]
        self.threshold = 0.1
        self.model_path = "models/model_unet_vgg_16_best.pt"
        self.model_type = "vgg16"
        self.out_pred_dir = None
        self.out_viz_dir = None
        self.crack_predict_results = "data/crack_results/crack_predict_results"
        self.crack_viz_results = "data/crack_results/crack_viz_results"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_tfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(self.channel_means, self.channel_stds)])

    def _evaluate_img(self, model, img):
        img_height, img_width, img_channels = img.shape
        input_width, input_height = input_size[0], input_size[1]

        img_1 = cv.resize(img, (input_width, input_height), cv.INTER_AREA)
        X = self.train_tfms(Image.fromarray(img_1))

        # Ensure the model is on CPU
        if self.device == torch.device('cpu'):
            X = Variable(X.unsqueeze(0)).cpu()  # [N, 1, H, W]
            model = model.cpu()
        else:
            X = Variable(X.unsqueeze(0)).cuda()  # [N, 1, H, W]

        mask = model(X)

        mask = F.sigmoid(mask[0, 0]).data.cpu().numpy()
        mask = cv.resize(mask, (img_width, img_height), cv.INTER_AREA)
        return mask

    def _disable_axis(self):
        plt.axis('off')
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.gca().axes.get_xaxis().set_ticklabels([])
        plt.gca().axes.get_yaxis().set_ticklabels([])

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

        if self.model_type == 'vgg16':
            model = load_unet_vgg16(self.model_path)
        elif self.model_type  == 'resnet101':
            model = load_unet_resnet_101(self.model_path)
        elif self.model_type  == 'resnet34':
            model = load_unet_resnet_34(self.model_path)
        else:
            raise ValueError(f"Model {self.model_type} is invalid!")

        paths = [path for path in Path(img_dir).glob('*.*')]
        raw_arr_imgs = []
        pred_arr_imgs = []
        for img_path in paths:
            img = Image.open(str(img_path))
            img = np.asarray(img)
            if len(img.shape) != 3:
                raise ValueError(f'incorrect image shape: {img_path.name}{img.shape}')
            
            if img.shape[0] > 2000 or img.shape[1] > 2000:
                img = cv.resize(img, None, fx=0.2, fy=0.2, interpolation=cv.INTER_AREA)

            img = img[:,:,:3]

            prob_map_full = self._evaluate_img(model, img)

            crack_mask = (prob_map_full * 255).astype(np.uint8)
            prob_map_viz_full = prob_map_full.copy()
            prob_map_viz_full = prob_map_viz_full/ prob_map_viz_full.max()
            prob_map_viz_full[prob_map_viz_full < self.threshold] = 0.0

            # convert to PIL image
            crack_mask_pil = Image.fromarray(crack_mask)
            img_pil = Image.fromarray(img)

            # add to array
            raw_arr_imgs.append(img)
            pred_arr_imgs.append(crack_mask)

            if self.out_pred_dir is not None:
                crack_mask_pil.save(join(self.out_pred_dir, f'{img_path.stem}_mask.jpg'))
                seg_results.append(join(self.out_pred_dir, f'{img_path.stem}_mask.jpg'))

            if self.out_viz_dir is not None:
                fig = plt.figure(figsize=(10, 5))
                fig.suptitle(f'Unet_VGG16 Model \n img={img_path.stem} \n threshold = {self.threshold}')
                ax = fig.add_subplot(131)
                ax.imshow(img_pil)
                ax = fig.add_subplot(132)
                ax.imshow(prob_map_viz_full)
                ax = fig.add_subplot(133)
                ax.imshow(img_pil)
                ax.imshow(prob_map_viz_full, alpha=0.4)
                plt.savefig(join(self.out_viz_dir, f'{img_path.stem}_viz.jpg'), dpi=500)
                plt.close('all')

                seg_results.append(join(self.out_viz_dir, f'{img_path.stem}_viz.jpg'))
            gc.collect()

        return seg_results, raw_arr_imgs, pred_arr_imgs