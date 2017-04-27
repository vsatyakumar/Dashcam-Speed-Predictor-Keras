import os
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Convolution2D, LSTM, Bidirectional, Activation, TimeDistributed, GlobalAveragePooling1D, Merge
from keras.regularizers import l1
from keras import applications
from keras.layers.normalization import BatchNormalization

# dimensions of our images.
#img_width, img_height = 224, 224

# fix random seed for reproducibility
np.random.seed(7)
#train_data_dir = '/input/data/images/train'
#validation_data_dir = '/input/data/images/validation'
nb_train_samples = 14280
nb_validation_samples = 6120
nb_batches_per_epoch=10
nb_epochs=50
instance_flag=1 #0 for loading data from Local, 1 for FloydHub instance
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
		batch_features=np.empty((timesteps, features_size,1))
		batch_labels=np.empty((timesteps,1))

		#print (batch_features.shape, batch_labels.shape )
		if flag == 0:
			index= np.random.randint(0, nb_train_samples-1, size = 1, dtype=np.int64)
		else:
			index= np.random.randint(0, nb_validation_samples-1, size = 1, dtype=np.int64)

		dataX = features[index,]
		dataY = labels[index,]
		#print dataX

		
		#print dataX.shape

		for j in range(1,timesteps):
			
			x = features[index-j,]
			y = labels[index-j,]
			#print x
			#x = np.expand_dims(x, axis=0)
			#y = np.expand_dims(x, axis=0)
			#print x.shape


			dataX = np.vstack((dataX,x))
			dataY= np.vstack((dataY,y))
			#print(dataX.shape)

		dataX=np.expand_dims(dataX, axis=-1)
		#print(dataX.shape)
		dataY=np.expand_dims(dataY, axis=-1)
		batch_features = np.dstack((batch_features,dataX))
		batch_labels = np.dstack((batch_labels,dataY))
		print (batch_features.shape, batch_labels.shape)

		count+=1
		if count==batch_size:
			#print("Batch Features Dims , Batch Labels Dims =" batch_features.input_shape, batch_labels.shape)
			print("Iteration number = ",count)
			count=0
			dataX=[]
			dataY=[]
			yield batch_features, batch_labels


#Buil Model
def buildmodel():
	model=Sequential()
	#model.add(TimeDistributed(Dense(256,activation='relu', kernel_initializer=initializers.glorot_uniform), input_shape=(None, timesteps, features_size)))
	#mode.add(TimeDistributed(BatchNormalization()))
	#model.add(TimeDistributed(Dense(128, activation='relu', kernel_initializer=initializers.glorot_uniform)))
	#model.add(TimeDistributed(BatchNormalization()))

	# model.add(LSTM(output_dim=255,return_sequences=True))

	left = Sequential()
	left.add(Bidirectional(LSTM(128, activation ='relu', return_sequences=True), input_shape=(timesteps,features_size)))
	left.add(Bidirectional(LSTM(128, activation = 'relu', return_sequences=False)))

	right = Sequential()
	right.add(Bidirectional(LSTM(128, activation='relu', return_sequences=True), input_shape=(timesteps,features_size)))
	right.add(Bidirectional(LSTM(128, activation='relu', return_sequences=False)))
	model.add(Merge([left, right], mode = 'concat'))
	model.add(Dense(100, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))
	model.add(Dense(1,activation='linear'))


	print('Compiling Model...')
	model.compile(optimizer='adam',
		loss='mse',
		metrics=['accuracy'])
	
	return model
#Load Bottleneck Features (Resnet50) & Labels
print('Loading Bottleneck Features Data and Labels...')

if instance_flag==0:
	train_data = np.load(open('data/bottleneck_features_train.npy'))
	validation_data = np.load(open('data/bottleneck_features_validation.npy'))
	labels = np.loadtxt('data/train.txt')
	labels.astype(float)
else:
	train_data = np.load(open('/input/bottleneck_features_train.npy'))
	validation_data = np.load(open('/input/bottleneck_features_validation.npy'))
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

#model = Sequential()
# Regression layer
#model.add(TimeDistributed(LSTM(units=128), input_shape=(None, timesteps, features_size)))
#model.add(TimeDistributed(Dense(units=1), input_shape=(None,timesteps)))
#model.add(GlobalAveragePooling1D())
#model.add(Activation('linear'))

print('Building Model...')

model = buildmodel()

print('Training...')

model.fit_generator(generator(X_train, y_train, batch_size, timesteps, 0),
	steps_per_epoch=batch_size, epochs=50, validation_data=generator(X_validation, y_validation, batch_size, timesteps, 1), validation_steps=nb_validation_samples)

print('Training Successful - Saving Weights...')

if instance_flag==0:
	model.save_weights('data/lstm_speed_model.h5')
else:
	model.save_weights('/output/lstm_speed_model.h5')
