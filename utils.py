import numpy as np
import cv2
import random
from glob import glob
import os


class SRDataGenerator(object):
    '''
    Data generator for super-resolution models
    '''
    def flow_from_directory(self, input_dir, scale_factor=2, batch_size=32, input_filename='*'):
        filenames = glob(os.path.join(input_dir, input_filename))

        hr_images = []
        lr_images = []

        while True:
            np.random.shuffle(filenames)

            for filename in filenames:
                img = self.load_image(filename)
                if img is None or img.shape[0] < 256 or img.shape[1] < 256:
                    continue

                patches = self.genrate_random_patches(img)
                hr_images += patches
                lr_images += [self.downscale_image(p) for p in patches]

                while len(hr_images) >= batch_size:
                    indices = np.random.permutation(batch_size)
                    yield np.array(hr_images[:batch_size]).astype('float32')[indices],\
                          np.array(lr_images[:batch_size]).astype('float32')[indices]

                    hr_images, lr_images = hr_images[batch_size:], lr_images[batch_size:]

    def load_image(self, filename):
        try:
            return cv2.imread(filename).astype(np.float32)
        except AttributeError:
            return None

    def genrate_random_patches(self, img, num=None, patch_size=256):
        height, width, _ = img.shape
        patches = []
        if num is None:
            num = random.randint(1, 3)

        for _ in range(num):
            patch_y, patch_x = random.randint(0, height - patch_size),\
                               random.randint(0, width - patch_size)
            patch = img[patch_y:patch_y + patch_size, patch_x:patch_x + patch_size]
            patches.append(patch)

        return patches

    def downsample_image(self, img):
        new_img = cv2.pyrDown(img)
        new_img = cv2.resize(new_img, (img.shape[1], img.shape[0]),
                             interpolation=cv2.INTER_CUBIC)
        return new_img


def apply_to_patch(img, fn, coord, patch_size, original_img=None, *args, **kwargs):
    '''
    Applies given function to a square patch of an image.
    original_img (optional): image from which to take the patch
    '''
    y, x = coord

    if original_img is not None:
        patch = original_img[y:y + patch_size,
                             x:x + patch_size].copy()
    else:
        patch = img[y:y + patch_size,
                    x:x + patch_size]

    patch = fn(patch, *args, **kwargs)

    img[y:y + patch_size,
        x:x + patch_size] = patch

    return img


def apply_patchwise(img, fn, patch_size=256, *args, **kwargs):
    '''
    Function that splits image into square patches and apples given function to every of them
    '''
    height, width, _ = img.shape
    original_img = img.copy()

    offset_height, offset_width = 0, 0

    while offset_height <= height - patch_size:
        while offset_width < width - patch_size:
            apply_to_patch(img, fn, (offset_height, offset_width), patch_size,
                           original_img=original_img, *args, **kwargs)
            offset_width += patch_size
        apply_to_patch(img, fn, (offset_height, width - patch_size), patch_size,
                       original_img=original_img, *args, **kwargs)

        offset_height += patch_size
        offset_width = 0

        if offset_height > height - patch_size and offset_height < height:
            offset_height = height - patch_size

    return img
