import os
import tarfile
import tempfile
from six.moves import urllib

import numpy as np
from PIL import Image

import tensorflow as tf


class DeepLabModel(object):
    '''
    DeepLab segmentation model
    '''

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, tarball_path):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()

        graph_def = None
        # Extract frozen graph from tar archive.
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
            if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.GraphDef.FromString(file_handle.read())
                break

        tar_file.close()

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.graph)

    def run(self, image):
        """Runs inference on a single image.
        Args:
            image: A PIL.Image object, raw input image.
        Returns:
            resized_image: RGB image resized from original input image.
            seg_map: Segmentation map of `resized_image`.
        """
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        return resized_image, seg_map

MODEL_NAME = 'mobilenet'

_DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
_MODEL_URLS = {
        'mobilenet':
                'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
        'xception':
                'deeplabv3_pascal_train_aug_2018_01_04.tar.gz',
}
_TARBALL_NAME = 'deeplab_model.tar.gz'

model_dir = tempfile.mkdtemp()
tf.gfile.MakeDirs(model_dir)

download_path = os.path.join(model_dir, _TARBALL_NAME)
print('downloading model, this might take a while...')
urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[MODEL_NAME],
                                     download_path)
print('download completed! loading DeepLab model...')

MODEL = DeepLabModel(download_path)
print('model loaded successfully!')




###============================================###
###--------------------------------------------###
###               RUNTIME PHASE                ###
###--------------------------------------------###
###============================================###



def run_visualization(filename):
    """Inferences DeepLab model and visualizes result."""
    original_im = Image.open(filename)

    print('running deeplab on image %s...' % filename)
        
    import time
    start = time.time()
    input_div = 1026
    if original_im.size[0] >= input_div:
        width, height = original_im.size
        target_width = width - width % input_div
        target_size = (target_width, int(height * target_width / width))
        resized_image = original_im.convert('RGB').resize(target_size, Image.ANTIALIAS)

        im = resized_image
        width, height = im.size

        tiles_num = ((width - 1) // input_div + 1, (height - 1) // input_div + 1)
        results = np.ndarray([4, width, height], np.int64)
        res_im = Image.new('RGB', (width, height))
        print(results.shape)
        for off in range(4):
            for i in range(tiles_num[0]):
                for j in range(tiles_num[1]):
                    box = [
                            i * input_div + (off & 1) * input_div // 2,
                            j * input_div + ((off & 2) >> 1) * input_div // 2,
                            min((i + 1) * input_div + (off & 1) * input_div // 2, width),
                            min((j + 1) * input_div + ((off & 2) >> 1) * input_div // 2, height)
                    ]
                    if not ((off & 1 and i == tiles_num[0] - 1) or (off & 2 and i == tiles_num[1] - 1)):
                            print(box)
                            tile = im.crop(box)
                            if 0 in tile.size:
                                    continue
                            print(tile.size)
                            img, seg_map = MODEL.run(tile)
                            w, h = seg_map.shape
                    
                            results[off][box[0]:box[0]+2*h:2,         box[1]:box[1]+2*w:2] = seg_map.T
                            results[off][box[0]+1:box[0]+2*h+1:2, box[1]:box[1]+2*w:2] = seg_map.T
                            results[off][box[0]:box[0]+2*h:2,         box[1]+1:box[1]+2*w+1:2] = seg_map.T
                            results[off][box[0]+1:box[0]+2*h+1:2, box[1]+1:box[1]+2*w+1:2] = seg_map.T
                    if off == 0:
                            res_im.paste(tile, box[0:2])
    else:
        im, res = MODEL.run(original_im)
        return im, res

    results = (results == 15).astype(np.int64)
    res = (results.sum(axis=0) > 0).astype(np.int64)
    print('Complete in %fs' % (time.time() - start))
    return im, res.T


def process(filename):
    im, res = run_visualization(filename)

    exp = im.copy()
    mask = Image.fromarray(255 - res.astype(np.uint8) * 255, 'L')
    black = Image.new('L', res.shape[::-1], 0)
    exp.paste(black, (0, 0), mask)

    filename = filename[:-4]    # remove .jpg
    new_filename = filename + 'SEGMENTED.png'
    exp.save(new_filename, 'PNG')
