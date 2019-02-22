
import numpy as np
import random
import json
from glob import glob
from keras.models import model_from_json,load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import  ModelCheckpoint,Callback,LearningRateScheduler
import keras.backend as K
from model import Unet_model
from losses import *


class SGDLearningRateTracker(Callback):

    
    def on_epoch_begin(self, epoch, logs = {}):
        optimizer = self.model.optimizer
        lr = K.get_value(optimizer.lr)
        decay = K.get_value(optimizer.decay)
        lr = lr / 10
        decay = decay * 10
        K.set_value(optimizer.lr, lr)
        K.set_value(optimizer.decay, decay)
        print("lr changed to:", lr)
        print("decay changed to:", decay)


class Training(object):


    '''
    load a model
    input: string 'model_name': filepath to model and weights, not including extension
    output: model with loaded weights. can fit on model using loaded_model = True in fit_model method
    '''
    def load_model(self, model_name):
        '''
        Load a model
        INPUT  (1) string 'model_name': filepath to model and weights, not including extension
        OUTPUT: Model with loaded weights. can fit on model using loaded_model=True in fit_model method
        '''
        print ('Loading model {}'.format(model_name))
        model_toload = '/Users/kaitlinylim/Documents/tumorproj/models/{}.json'.format(model_name)
        weights = '/Users/kaitlinylim/Documents/tumorproj/weights/{}.hdf5'.format(model_name)
        with open(model_toload) as f:
            loaded_model_json = f.read()
        model_comp = model_from_json(loaded_model_json)
        model_comp.load_weights(weights)
        print ('model loaded.')
        model_comp.compile(loss = gen_dice_loss, optimizer = SGD(lr = 0.08, momentum = 0.9, decay = 5e-6, nesterov = False),
                           metrics = [dice_whole_metric, dice_core_metric, dice_en_metric])
        print('model compiled')
        self.model = model_comp
        return model_comp


    # constructor
    def __init__(self, batch_size, nb_epoch, load_model_resume_training = None):
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch

        #loading model from path to resume previous training without recompiling the whole model
        if load_model_resume_training is not None:
            # self.model = self.load_model(load_model_resume_training,custom_objects={'gen_dice_loss': gen_dice_loss,'dice_whole_metric':dice_whole_metric,'dice_core_metric':dice_core_metric,'dice_en_metric':dice_en_metric})
            self.model = self.load_model(load_model_resume_training)
            print("pre-trained model loaded!")
        else:
            unet = Unet_model(img_shape = (128, 128, 4))
            self.model = unet.model
            print("u-net cnn compiled!")


    def fit_unet(self, X33_train, Y_train, X_patches_valid = None, Y_labels_valid = None):
        train_generator = self.img_msk_gen(X33_train, Y_train, 9999)
        checkpointer = ModelCheckpoint(filepath = "/Users/kaitlinylim/Documents/tumorproj/checkpoints/ResUnet.{epoch:02d}_{val_loss:.3f}.hdf5",
                                      verbose = 1)
        self.model.fit_generator(train_generator, steps_per_epoch = len(X33_train)//self.batch_size,
                                epochs = self.nb_epoch, validation_data = (X_patches_valid, Y_labels_valid),
                                verbose = 1, callbacks = [checkpointer, SGDLearningRateTracker()])


    '''
    a custom generator that performs data augmentation on both patches and their corresponding masks
    '''
    def img_msk_gen(self, X33_train, Y_train, seed):
        datagen = ImageDataGenerator(horizontal_flip = True, data_format = 'channels_last')
        datagen_msk = ImageDataGenerator(horizontal_flip = True, data_format = 'channels_last')
        image_generator = datagen.flow(X33_train, batch_size = 4, seed = seed)
        y_generator = datagen_msk.flow(Y_train, batch_size = 4, seed = seed)
        while True:
            yield(image_generator.next(), y_generator.next())


    '''
    input: string 'model_name': path where to save model and weights without extension
    saves current model as json and weights as h5df file
    '''
    def save_model(self, model_name):
        model_tosave = '{}.json'.format(model_name)
        json_string = self.model.to_json()
        with open('/Users/kaitlinylim/Documents/tumorproj/models/'+model_tosave, 'w') as json_file:
            json_file.write(json_string)
            weights = '{}.hdf5'.format(model_name)
            self.model.save_weights('/Users/kaitlinylim/Documents/tumorproj/weights/'+weights)
        print ('model saved')


if __name__ == '__main__':

    X_patches = np.load('/Users/kaitlinylim/Documents/tumorproj/numpy/X_patches.npy')
    Y_labels = np.load('/Users/kaitlinylim/Documents/tumorproj/numpy/Y_labels.npy')
    print('loading patches done\n')

    # split the training data further into training data and validation data
    div = int(X_patches.shape[0] * 0.8)
    X_training = X_patches[0:div]
    X_validation = X_patches[div:]
    Y_training = Y_labels[0:div]
    Y_validation = Y_labels[div:]

    brain_seg_1 = Training(batch_size = 4, nb_epoch = 2)

    print("number of trainable parameters:", brain_seg_1.model.count_params())
    # print(brain_seg_1.model.summary())

    brain_seg_1.fit_unet(X_training, Y_training, X_validation, Y_validation)

    # if an already trained model is being loaded
    model_to_load = 'seg_2_epochs'
    brain_seg_2 = Training(batch_size = 4, nb_epoch = 2, load_model_resume_training = model_to_load)
    brain_seg_2.fit_unet(X_training, Y_training, X_validation, Y_validation)
    brain_seg_2.save_model('seg_4_epochs')
