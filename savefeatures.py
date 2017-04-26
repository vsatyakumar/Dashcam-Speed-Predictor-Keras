#KERAS_BACKEND=tensorflow python -c "from keras import backend"
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Convolution2D
from keras import applications
# dimensions of our images.
img_width, img_height = 224, 224

train_data_dir = '/input/data/images/train'
validation_data_dir = '/input/data/images/validation'
nb_train_samples = 14280
nb_validation_samples = 6120
batch_size = 1



def save_bottlebeck_features():
    
    datagen = ImageDataGenerator(#featurewise_center=False,
        #samplewise_center=False,
        #featurewise_std_normalization=False,
        #samplewise_std_normalization=False,
        #zca_whitening=True,
        #rotation_range=0.,
        #width_shift_range=0.,
        #height_shift_range=0.,
    #     shear_range=0.,
    #     zoom_range=0.,
    #     channel_shift_range=0.,
    #     fill_mode='nearest',
    #     cval=0.,
         rescale=1./255,
    #    dim_ordering=False,
        #horizontal_flip=False,
        )
    
    # build the VGG16 network
    #model = applications.VGG16(include_top=False, weights='imagenet')
    model = applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=None)

    #Create Batch Generators
    train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(train_generator, nb_train_samples)
    np.save(open('/output/bottleneck_features_train.npy', 'wb'), bottleneck_features_train)

    validation_generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(validation_generator, nb_validation_samples)
    np.save(open('/output/bottleneck_features_validation.npy', 'wb'), bottleneck_features_validation)

    return bottleneck_features_validation, bottleneck_features_train

x_validation, x_train = save_bottlebeck_features()