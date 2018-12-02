from keras.models import Sequential
from keras.layers import Conv2D
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
import numpy as np
import subprocess
import os
from glob import glob

from utils import SRDataGenerator, apply_patchwise, upscale_image

SRCNN_WEIGHTS_PATH = 'base_srcnn.h5'
SRCNN_INPUT_SHAPE = (256, 256, 3)
SRCNN_LEARNING_RATE = 3e-4
SRCNN_BATCH_SIZE = 32

W2X_BASEDIR = 'waifu2x'
W2X_BASE_MODEL_NAME = 'photo'

DEFAULT_CALLBACKS = [ModelCheckpoint('best_weights.h5',
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=True),
                     ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
                     EarlyStopping(monitor='val_loss', patience=10)]


class SRCNN(object):
    def __init__(self,
                 weights_path=SRCNN_WEIGHTS_PATH,
                 input_shape=SRCNN_INPUT_SHAPE,
                 learning_rate=SRCNN_LEARNING_RATE):
        self.model = self.create_model(input_shape, learning_rate)
        self.weights_path = weights_path

    def create_model(self, input_shape, learning_rate):
        '''
        SRCNN model as proposed by Dong et al.
        '''
        model = Sequential()
        model.add(Conv2D(filters=128, kernel_size=(9, 9),
                         init='glorot_uniform',
                         activation='relu',
                         border_mode='same',
                         input_shape=input_shape))
        model.add(Conv2D(filters=64, kernel_size=(3, 3),
                         init='glorot_uniform',
                         activation='relu',
                         border_mode='same',
                         bias=True))
        model.add(Conv2D(filters=3, kernel_size=(5, 5),
                         init='glorot_uniform',
                         activation='linear',
                         border_mode='same',
                         bias=True))

        adam = Adam(lr=learning_rate)
        model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])

        return model

    def load_weights(self):
        self.model.load_weights(self.weights_path)

    def fit(self, train_dir='train_data', epochs=2, batch_size=SRCNN_BATCH_SIZE,
            val_dir=None, callbacks=DEFAULT_CALLBACKS, restore_weights_from=None):
        '''
        Train the model. If validation directory is provided, callbacks are applied
        '''
        train_filenames = glob(os.path.join(train_dir, '*'))

        data_generator = SRDataGenerator()
        train_generator = data_generator.flow_from_directory(train_dir, batch_size=batch_size)

        if val_dir is not None:
            val_filenames = glob(os.path.join(val_dir, '*'))
            val_generator = data_generator.flow_from_directory(val_dir, batch_size=batch_size)

            self.model.fit_generator(
                generator=train_generator,
                steps_per_epoch=int(len(train_filenames) * 2.0 / batch_size),
                validation_data=val_generator,
                validation_steps=int(len(val_filenames) * 2.0 / batch_size),
                epochs=epochs,
                callbacks=callbacks)

            if restore_weights_from is not None:
                self.model.load_weights(restore_weights_from)

        else:
            self.model.fit_generator(
                generator=train_generator,
                steps_per_epoch=int(len(train_filenames) * 2.0 / batch_size),
                epochs=epochs)

    def process_patch(self, patch):
        return (self.model.predict(np.array([patch / 255.0]))[0] * 255.0).astype(np.int32)

    def enhance(self, img, upscale=True):
        if upscale:
            img = upscale_image(img, 2)

        return apply_patchwise(img, self.process_patch)


class W2X(object):
    '''
    Waifu2x models wrapper
    '''

    def __init__(self, basedir=W2X_BASEDIR, base_model_name=W2X_BASE_MODEL_NAME):
        self.basedir = basedir
        self.base_model_path = os.path.join(basedir, 'models', base_model_name)

    def denoise(self, image_path, result_path, noise_level=1, model_name=W2X_BASE_MODEL_NAME):
        subprocess.check_output(['th',
                                 os.path.join(self.basedir, 'waifu2x.lua'),
                                 '-model_dir', os.path.join(self.basedir, 'models', model_name),
                                 '-m', 'noise',
                                 '-noise_level', str(noise_level),
                                 '-i', image_path,
                                 '-o', result_path,
                                 '-force_cudnn', '1'])

    def upscale(self, image_path, result_path, model_name=W2X_BASE_MODEL_NAME):
        subprocess.check_output(['th',
                                 os.path.join(self.basedir, 'waifu2x.lua'),
                                 '-model_dir', os.path.join(self.basedir, 'models', model_name),
                                 '-m', 'scale',
                                 '-i', image_path,
                                 '-o', result_path,
                                 '-force_cudnn', '1'])

    def enchance(self, image_path, result_path, noise_level=1, model_name=W2X_BASE_MODEL_NAME):
        subprocess.check_output(['th',
                                 os.path.join(self.basedir, 'waifu2x.lua'),
                                 '-model_dir', os.path.join(self.basedir, 'models', model_name),
                                 '-m', 'noise_scale',
                                 '-noise_level', str(noise_level),
                                 '-i', image_path,
                                 '-o', result_path,
                                 '-force_cudnn', '1'])
