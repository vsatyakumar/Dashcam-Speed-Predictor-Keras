import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Convolution2D, LSTM, merge, Bidirectional, Activation
import os
from keras import applications
# dimensions of our images.
#img_width, img_height = 224, 224

#train_data_dir = '/input/data/images/train'
#validation_data_dir = '/input/data/images/validation'
nb_train_samples = 14280
nb_validation_samples = 6120
nb_batches_per_epoch=10
nb_epochs=50

#MAIN
nb_train_samples = 14280
nb_validation_samples = 6120
lstm_num_timesteps=10 #How many timesteps of features are being fed per sample of a batch. 


batch_size=10 #Samples per batch.
timesteps=10
features_size=2048

#Define Generator


def generator(features, labels, batch_size, timesteps, flag=0):

	while True:
		count=0
		print("Iteration number = ",count)
		batch_features=np.empty((1,timesteps, features_size))
		batch_labels=np.empty((1,timesteps))

		#print (batch_features.shape, batch_labels.shape )
		if flag==0:
			index= np.random.randint(0, nb_train_samples, size = 1, dtype=np.int64)
		else:
			index= np.random.randint(0, nb_validation_samples, size = 1, dtype=np.int64)

			dataX = features[index,]
			dataY = labels[index,]

			for j in range(1,timesteps):
				y = labels[index-j,]
				x= features[index-j,]

				dataX=np.append(x, dataX, axis=0)
				dataY= np.append(y, dataY, axis=0)
			
			dataX=np.expand_dims(dataX, axis=0)
			dataY=np.expand_dims(dataY, axis=0)
			
			batch_features=np.append(dataX, batch_features, axis=0)
			batch_labels=np.append(dataY, batch_labels, axis=0)

			count+=1

			if count==batch_size:
				print("Batch Features Dims , Batch Labels Dims =" batch_features.input_shape, batch_labels.shape)
				count=0
				yield batch_features, batch_labels


#Load Bottleneck Features (Resnet50) & Labels

print('Loading Bottleneck Feature Data...')
train_data = np.load(open('/input/bottleneck_features_train.npy'))
validation_data = np.load(open('/input/bottleneck_features_validation.npy'))

#Load Labels
print('Loading Labels...')
labels = np.loadtxt('/input/train.txt')
labels.astype(float)
y_train= labels[0:nb_train_samples]
y_validation=labels[nb_train_samples:len(labels)]


#print keras.backend.shape(train_data)
#print keras.backend.shape(validation_data)

X_train=np.reshape(train_data, (nb_train_samples, -1))
X_validation=np.reshape(validation_data, (nb_validation_samples, -1))

X_train.astype(float)
X_validation.astype(float)

print('Final Data Shape = [xtrain, xvalidation, ytain, yvalidation]', X_train.shape , X_validation.shape, y_train.shape, y_validation.shape)

model = Sequential()
model.add(LSTM(256, return_sequences=True, input_shape=(timesteps, features_size)))
model.add(LSTM(10, return_sequences=True, input_shape=(timesteps,256)))

# Regression layer

model.add(Dense(units=256))
model.add(Activation('sigmoid'))
model.add(Dense(units=1))
model.add(Activation('linear'))

print('Compiling Model...')
model.compile(optimizer='adam',
	loss='mse',
	metrics=['accuracy'])

print('Training...')

model.fit_generator(generator(X_train, y_train, batch_size, timesteps, 0),
	steps_per_epoch=nb_train_samples, epochs=50, validation_data=generator(X_validation, y_validation, batch_size, timesteps, 1), validation_steps=nb_validation_samples)

print('Training Successful - Saving Weights...')
model.save_weights('/output/lstm_speed_model.h5')

