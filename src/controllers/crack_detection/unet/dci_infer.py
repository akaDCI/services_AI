import os
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

        self.train_tfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(self.channel_means, self.channel_stds)])

    def _evaluate_img(self, model, img):
        img_height, img_width, img_channels = img.shape
        input_width, input_height = input_size[0], input_size[1]

        img_1 = cv.resize(img, (input_width, input_height), cv.INTER_AREA)
        X = self.train_tfms(Image.fromarray(img_1))
        X = Variable(X.unsqueeze(0)).cuda()  # [N, 1, H, W]

        mask = model(X)

        mask = F.sigmoid(mask[0, 0]).data.cpu().numpy()
        mask = cv.resize(mask, (img_width, img_height), cv.INTER_AREA)
        return mask

    def _evaluate_img_patch(self, model, img):
        input_width, input_height = input_size[0], input_size[1]

        img_height, img_width, img_channels = img.shape

        if img_width < input_width or img_height < input_height:
            return self._evaluate_img(model, img)

        stride_ratio = 0.1
        stride = int(input_width * stride_ratio)

        normalization_map = np.zeros((img_height, img_width), dtype=np.int16)

        patches = []
        patch_locs = []
        for y in range(0, img_height - input_height + 1, stride):
            for x in range(0, img_width - input_width + 1, stride):
                segment = img[y:y + input_height, x:x + input_width]
                normalization_map[y:y + input_height, x:x + input_width] += 1
                patches.append(segment)
                patch_locs.append((x, y))

        patches = np.array(patches)
        if len(patch_locs) <= 0:
            return None

        preds = []
        for i, patch in enumerate(patches):
            patch_n = self.train_tfms(Image.fromarray(patch))
            X = Variable(patch_n.unsqueeze(0)).cuda()  # [N, 1, H, W]
            masks_pred = model(X)
            mask = F.sigmoid(masks_pred[0, 0]).data.cpu().numpy()
            preds.append(mask)

        probability_map = np.zeros((img_height, img_width), dtype=float)
        for i, response in enumerate(preds):
            coords = patch_locs[i]
            probability_map[coords[1]:coords[1] + input_height, coords[0]:coords[0] + input_width] += response

        return probability_map

    def _disable_axis(self):
        plt.axis('off')
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.gca().axes.get_xaxis().set_ticklabels([])
        plt.gca().axes.get_yaxis().set_ticklabels([])

    def infer(self, img_folder):
        img_dir = f"tmp/upload_files/{img_folder}"
        model_path = "src/models/model_unet_vgg_16_best.pt"
        model_type = "vgg16"
        crack_predict_results = "data/crack_results/crack_predict_results"
        out_pred_dir = f"{crack_predict_results}/{img_folder}"
        crack_viz_results = "data/crack_results/crack_viz_results"
        out_viz_dir = f"{crack_viz_results}/{img_folder}"
        threshold = 5

        if out_viz_dir != '':
            for path in Path(crack_viz_results).glob('*'):
                shutil.rmtree(str(path))
            os.makedirs(out_viz_dir, exist_ok=True)

        if out_pred_dir != '':
            for path in Path(crack_predict_results).glob('*'):
                shutil.rmtree(str(path))
            os.makedirs(out_pred_dir, exist_ok=True)

        if model_type == 'vgg16':
            model = load_unet_vgg16(model_path)
        elif model_type  == 'resnet101':
            model = load_unet_resnet_101(model_path)
        elif model_type  == 'resnet34':
            model = load_unet_resnet_34(model_path)
            print(model)
        else:
            print('undefind model name pattern')
            exit()

        paths = [path for path in Path(img_dir).glob('*.*')]
        raw_imgs = []
        pred_imgs = []
        for img_path in paths:
            img_0 = Image.open(str(img_path))
            img_0 = np.asarray(img_0)
            if len(img_0.shape) != 3:
                print(f'incorrect image shape: {img_path.name}{img_0.shape}')
            
            if img_0.shape[0] > 2000 or img_0.shape[1] > 2000:
                img_0 = cv.resize(img_0, None, fx=0.2, fy=0.2, interpolation=cv.INTER_AREA)

            img_0 = img_0[:,:,:3]

            prob_map_full = self._evaluate_img(model, img_0)

            pred_img = (prob_map_full * 255).astype(np.uint8)

            # add to array
            raw_imgs.append(img_0)
            pred_imgs.append(pred_img)

            if out_pred_dir != '':
                cv.imwrite(filename=join(out_pred_dir, f'{img_path.stem}.jpg'), img=pred_img)

            if out_viz_dir != '':
                img_1 = img_0
                prob_map_patch = self._evaluate_img_patch(model, img_1)

                prob_map_viz_patch = prob_map_patch.copy()
                prob_map_viz_patch = prob_map_viz_patch/ prob_map_viz_patch.max()
                # prob_map_viz_patch[prob_map_viz_patch < threshold] = 0.0

                fig = plt.figure()
                st = fig.suptitle(f'name={img_path.stem} \n cut-off threshold = {threshold}', fontsize="x-large")
                ax = fig.add_subplot(231)
                ax.imshow(img_1)
                ax = fig.add_subplot(232)
                ax.imshow(prob_map_viz_patch)
                ax = fig.add_subplot(233)
                ax.imshow(img_1)
                ax.imshow(prob_map_viz_patch, alpha=0.4)

                plt.savefig(join(out_viz_dir, f'{img_path.stem}.jpg'), dpi=500)
                plt.close('all')

            gc.collect()

        return raw_imgs, pred_imgs