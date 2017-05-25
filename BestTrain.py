import os
import numpy as np
import tensorflow as tf
import cv2
import argparse
import keras
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from keras.models import Sequential, Model
from keras.models import model_from_json
from keras.optimizers import RMSprop, SGD, Adam
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.core import Reshape
from keras.layers import Dropout, Flatten, Dense, Conv2D, LSTM, Merge, Bidirectional, Activation, TimeDistributed, GRU, MaxPooling2D, Input, BatchNormalization, GlobalAveragePooling2D, SeparableConv2D, Cropping2D, Lambda
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, History
from random import randint, shuffle
from keras import backend as K
from keras.regularizers import l1_l2
from keras import losses
from keras.applications.inception_v3 import InceptionV3
import random

K.set_image_dim_ordering('tf')
np.random.seed(1337)
learning_rate = 0.01
img_width, img_height = 250,250
nb_train_samples = 14280
nb_validation_samples = 6120
train_batch_size=1
validation_batch_size=1
val_steps=100
train_steps=100
nb_epochs=200
channels=3 #RGB
timesteps=8
frequency=1 #Gap between successive frames being fed to the model. Not to be confused with FPS (Which is 20, if you're wondering)
fps=20

#Define hyperparameters
weight_init='glorot_uniform'
stateful=False
#Act =LeakyReLU(alpha=0.3)
Act1=Activation('softplus')
#Act2=Activation('relu')
#Act1=LeakyReLU(alpha=5.0)
#Act4=PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)

l1_l2_conv=0.1
l1_l2_gru=0.1
l1_l2_dense=0.1


instance_flag=1 #0 for loading data from Local, 1 for FloydHub instance

if instance_flag==0:
    root_dir, _ = os.path.split(os.path.abspath(__file__))
    train_root_dir = root_dir + '/data/images/train'
    val_root_dir = root_dir + '/data/images/validation'
    speeds = np.loadtxt(root_dir + '/data/train.txt')
else:
    train_root_dir =  '/input/images/train'
    val_root_dir = '/input/images/validation'
    speeds = np.loadtxt('/input/train.txt')

def train():
    print('Building Model...')
    model = buildmodel(summary=1)
    
    if instance_flag==0:
        print("Saving Model to disk...")
            #Serialize model to JSON
        model_json = model.to_json()
        with open("./model/speed-predictor-keras-model-v1.json", "w") as json_file:
            json_file.write(model_json)
        checkpointer = ModelCheckpoint(filepath="./weights/dashcam_weights-{epoch:02d}-{val_mean_squared_error:.2f}.hdf5", verbose=1, save_best_only=True)
        tensorboard = TensorBoard(log_dir='./logs/', histogram_freq=0, write_graph=True, write_images=False)
    else:
        print("Saving Model to disk...")
        #Serialize model to JSON
        model_json = model.to_json()
        with open("/output/speed-predictor-keras-model-v1.json", "w") as json_file:
            json_file.write(model_json)
        checkpointer = ModelCheckpoint(filepath="/output/dashcam_weights-{epoch:02d}-{val_mean_squared_error:.2f}.hdf5", verbose=1, save_best_only=True)
        tensorboard = TensorBoard(log_dir='/output/', histogram_freq=0, write_graph=True, write_images=False)

    y_train= speeds[0:nb_train_samples]
    y_validation= speeds[nb_train_samples:len(speeds)]

    y_train=np.expand_dims(y_train, axis=-1)
    y_validation=np.expand_dims(y_train, axis=-1)

    train_generator = generator(y_train, 0)
    val_generator = generator(y_validation, 1)

    earlyStopping= EarlyStopping(monitor='val_mean_squared_error', patience=3, verbose=2, mode='auto')

    print('Training...')
    
    model.fit_generator(train_generator, steps_per_epoch=train_steps, epochs=nb_epochs, verbose=1, validation_data=val_generator, 
        validation_steps=val_steps, callbacks=[tensorboard, earlyStopping, checkpointer])

    print('We are done! Hope that was a terrific training sesh... :)')

def buildmodel(summary):

    data= Input(shape=(None,img_width,img_height,channels)) 
    convs= Sequential()
    convs.add(Lambda(lambda x: x/127.5 - 1.0,input_shape=(img_width, img_height, channels)))
    convs.add(Cropping2D(cropping=((50,10),(10,10))))
    convs.add(Conv2D(16, (3, 3), strides=(2,2),kernel_initializer=weight_init, activity_regularizer=l1_l2(l1_l2_conv), padding='same', bias=False, activation='elu',name="conv_1_1"))
    convs.add(MaxPooling2D((3, 3), strides=(2,2), padding='same'))
    convs.add(Conv2D(16, (3, 3), strides=(2,2) , kernel_initializer=weight_init, activity_regularizer=l1_l2(l1_l2_conv), padding='same', bias=False,activation='elu',name="conv_1_2"))
    convs.add(MaxPooling2D((3, 3), strides=(1,1), padding='same'))
    convs.add(GlobalAveragePooling2D())
    
    residual1 = Sequential()
    residual1.add(convs)
    residual1.pop()
    residual1.add(Conv2D(32, (3, 3), strides=(2,2),kernel_initializer=weight_init, activity_regularizer=l1_l2(l1_l2_conv), padding='same', bias=False, activation='elu',name="conv_2_1", input_shape=(img_width, img_height, channels)))
    residual1.add(MaxPooling2D((2 ,2), strides=(1,1), padding='same'))
    residual1.add(Conv2D(32, (3, 3), strides=(2,2),kernel_initializer=weight_init, activity_regularizer=l1_l2(l1_l2_conv), padding='same', bias=False, activation='elu',name="conv_2_2", input_shape=(img_width, img_height, channels)))
    residual1.add(MaxPooling2D((3, 3), strides=(1,1), padding='same'))
    residual1.add(GlobalAveragePooling2D())

    residual2 = Sequential()
    residual2.add(residual1)
    residual2.pop()
    residual2.add(Conv2D(64, (4, 4), strides=(2,2),kernel_initializer=weight_init, activity_regularizer=l1_l2(l1_l2_conv), padding="same", bias=False, activation='elu',name="conv_3_1", input_shape=(img_width, img_height, channels)))
    residual2.add(MaxPooling2D((2, 2), strides=(1,1), padding='same'))
    residual2.add(Conv2D(64, (4, 4), strides=(2,2),kernel_initializer=weight_init, activity_regularizer=l1_l2(l1_l2_conv), padding="same", bias=False, activation='elu',name="conv_3_2", input_shape=(img_width, img_height, channels)))
    residual2.add(MaxPooling2D((2, 2), strides=(1,1), padding='same'))
    residual2.add(GlobalAveragePooling2D())

    residual3 = Sequential()
    residual3.add(residual2)
    residual3.pop()
    residual3.add(Conv2D(128, (3, 3), strides=(2, 2),kernel_initializer=weight_init, activity_regularizer=l1_l2(l1_l2_conv), padding='same', bias=False, activation='elu',name="conv_4_1", input_shape=(img_width, img_height, channels)))
    residual3.add(MaxPooling2D((3, 3), strides=(1,1), padding='same'))
    residual3.add(Conv2D(128, (3, 3), strides=(2,2),kernel_initializer=weight_init, activity_regularizer=l1_l2(l1_l2_conv), padding='same', bias=False, activation='elu',name="conv_4_2"))
    residual3.add(MaxPooling2D((3, 3), strides=(1,1), padding='same'))
    residual3.add(GlobalAveragePooling2D())

    resi1 = TimeDistributed(residual1)(data)
    resi2 = TimeDistributed(residual2)(data)
    resi3 = TimeDistributed(residual3)(data)
    out = TimeDistributed(convs)(data)

    #---------------RECURRENT LAYERS-------------------------------------------------------------

    x1 = Bidirectional(GRU(64, activation='elu', kernel_initializer=weight_init, recurrent_activation='' ,return_sequences=False, name='gru_1_0', activity_regularizer=l1_l2(l1_l2_gru) ))(out)
    x1 = Act1(x1)

    out1 = Bidirectional(GRU(64, activation='elu', kernel_initializer=weight_init, activity_regularizer=l1_l2(l1_l2_gru),recurrent_activation='elu', return_sequences=False, name='gru_1_1'))(resi1)  
    out1= Act1(out1)
    out2 = Bidirectional(GRU(64, activation='elu', kernel_initializer=weight_init, activity_regularizer=l1_l2(l1_l2_gru),recurrent_activation='elu', return_sequences=False, name='gru_1_2'))(resi2)   
    out2= Act1(out2)
    out3 = Bidirectional(GRU(64, activation='elu', kernel_initializer=weight_init , activity_regularizer=l1_l2(l1_l2_gru),recurrent_activation='elu', return_sequences=False, name='gru_1_3'))(resi3)  
    out3= Act1(out3)
    #x_2 = Act2(x1)
    #x_3 = Act3(x1)
    #x_4=Act1(x1)
    #Some interesting permutations and combinations that worked to achieve the smallest loss, so far...
    left = keras.layers.average([x1,out1])
    centre = keras.layers.average([left,out2])
    right = keras.layers.average([centre,out3])
    #top = keras.layers.average([x1,out4])
    

    xA = keras.layers.concatenate([left, centre])
    xB = keras.layers.concatenate([centre, right])
    xC = keras.layers.concatenate([right, left])
    #xD = keras.layers.maximum([left, top])
    
    x = keras.layers.concatenate([xA, xB, xC])
    
    #--------------------DENSE LAYERS-----------------------------------------------------------------------------
    speed = Dense(512, kernel_initializer=weight_init, activity_regularizer=l1_l2(l1_l2_dense), name='dense_1')(x)
    speed = Dropout(0.5)(speed)
    speed = Activation('elu')(speed)

    #speed=BatchNormalization()(x)
    speed = Dense(512, kernel_initializer=weight_init, activity_regularizer=l1_l2(l1_l2_dense), name='dense_2')(speed)
    speed = Dropout(0.2)(speed)
    speed = Activation('elu')(speed)
    speed= Dense(1, kernel_initializer=weight_init, activation='linear', name='speed')(speed)

    model = Model(inputs=data, outputs=speed)

    decay_rate = learning_rate / nb_epochs
    #sgd = SGD(lr=learning_rate, decay=decay_rate, momentum=0.999, nesterov=True, clipnorm=0.3)
    #optimize = Adam(lr = learning_rate, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08, decay = decay_rate)
    optimize = RMSprop(lr= learning_rate, decay = decay_rate)
    
    print('Compiling Model...')
    model.compile(optimizer = optimize,
        loss = l1_smooth_loss,
        metrics = ['mse'])
    
    if summary:
        print(model.summary())
        return model

def preprocess_img(img, flip=0):
    # central square crop
    min_side = min(img.shape[:-1])
    centre = img.shape[0]//2, img.shape[1]//2
    img = img[centre[0]-min_side//2:centre[0]+min_side//2,
              centre[1]-min_side//2:centre[1]+min_side//2,
              :]
    # rescale to standard size
    img = cv2.resize(img,(img_width, img_height), interpolation = cv2.INTER_AREA)
    #img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if flip==1:
        img=cv2.flip(img,1)

    #NOTE-------------CLAHE hurts performance!--------------------------------------------------
    #Reference for the claim (and I can vouch for it. ) : https://arxiv.org/pdf/1606.02228v2.pdf
    #"Quote: Global [42] and local(CLAHE [43]) histogram equalizations hurt performance as well"
    

    #img=np.array(img, dtype=np.uint8)
    #lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    #l, a, b = cv2.split(lab)
    #-----Applying CLAHE to L-channel---------------------------
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #cl = clahe.apply(l)
    #limg = cv2.merge((cl,a,b))
    #img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    #----------------------------------------------
    
    img= np.array(img, dtype='float32')
    #img = np.swapaxes(img,0,1)
    img=np.expand_dims(img,axis=0)
    # roll color axis to axis 0
    #img = np.swapaxes(img,0,1)
    return img

#Define Smooth L1 Loss
def l1_smooth_loss(y_true, y_pred):
    abs_loss = tf.abs(y_true - y_pred)
    sq_loss = 0.5 * (y_true - y_pred)**2
    l1_loss = tf.where(tf.less(abs_loss, 1.0), sq_loss, abs_loss - 0.5)
    return tf.reduce_sum(l1_loss, -1)    

def generator(labels, flag):
    count=0
    dataX=[]
    dataY=[]
    batch_features=[]
    batch_labels=[]
    #timesteps = np.random.randint(3,8)

    train_index_pointer = np.linspace(100,nb_train_samples-1, nb_train_samples, dtype=np.int64)
    train_index_pointer= np.random.permutation(train_index_pointer)
    train_index_pointer = list(train_index_pointer)

    val_index_pointer = list(np.linspace(100,nb_validation_samples-1, nb_validation_samples, dtype=np.int64))
    val_index_pointer= np.random.permutation(val_index_pointer)
    val_index_pointer = list(val_index_pointer)    

    #print('Generator Active')

    
    while True:

        if flag == 0:
            batch_size=train_batch_size
            trainbag=len(train_index_pointer)
            if trainbag<=100:
                #print("Train Data Exhausted - Resetting for further epochs...")
                train_index_pointer = np.linspace(100,nb_train_samples-1, nb_train_samples, dtype=np.int64)
                train_index_pointer= np.random.permutation(train_index_pointer)
                train_index_pointer = list(train_index_pointer)
                trainbag=len(train_index_pointer)
                ind = train_index_pointer.pop(trainbag-1)
                
                #random flip the sequence
                flip = np.random.randint(0,1)
            else:
                ind = train_index_pointer.pop(trainbag-2*np.random.randint(10,40))
                flip = np.random.randint(0,1)
            
            index = '{:0>5}'.format(ind)
            dataX = cv2.imread(train_root_dir + '/frame-' + index + '.jpeg',3)
            dataX =  preprocess_img(dataX,flip)
            dataY = labels[ind]
        else:
            batch_size=validation_batch_size
            validationbag=len(val_index_pointer)
            if validationbag<=400:
                #print("Validation Data Exhausted - Resetting for further epochs...")
                val_index_pointer = list(np.linspace(100,nb_validation_samples-1, nb_validation_samples, dtype=np.int64))
                val_index_pointer= np.random.permutation(val_index_pointer)
                val_index_pointer = list(val_index_pointer)  

                validationbag=len(train_index_pointer)
                ind = val_index_pointer.pop(validationbag-1)
                flip = np.random.randint(0,1)
            else:
                ind = val_index_pointer.pop(validationbag-2*np.random.randint(10,40))
                flip = np.random.randint(0,1)
            
            index = ind + nb_train_samples
            index='{:0>5}'.format(index)
            dataX = cv2.imread(val_root_dir + '/frame-' + index + '.jpeg',3)
            dataX=preprocess_img(dataX,flip)
            dataY = labels[ind]


        #Create One sample of (timesteps, features_size)
        for j in range(1,timesteps):
            
            pointer = ind-(j*frequency)
            if pointer<0:
               pointer = nb_train_samples + pointer
            
            #If using Validation, Reformat pointer according to filenames.
            if flag==1:
                pointer = pointer + nb_train_samples
                pointstr='{0:0>5}'.format(pointer)
                img = cv2.imread(val_root_dir + '/frame-' + pointstr + '.jpeg',3)
                img=preprocess_img(img, flip)

            else:
                pointstr='{0:0>5}'.format(pointer)
                img = cv2.imread(train_root_dir + '/frame-' + pointstr + '.jpeg',3)
                img=preprocess_img(img, flip)
            
            
            dataX=np.append(dataX, img, axis=0)
            #print("datax shape=" , dataX.shape)

        X_train=np.expand_dims(dataX, axis=0)
        
        #Stack samples to create batch
        if count==0:
            batch_features = X_train
            batch_labels=dataY
        else:
            batch_features=np.append(batch_features, X_train, axis=0)
            batch_labels = np.append(batch_labels, dataY, axis=0)
        
        count+=1
        if count == batch_size:
            count=0
            yield batch_features, batch_labels

train()



