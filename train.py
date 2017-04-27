import os
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Convolution2D, LSTM, Bidirectional, Activation, TimeDistributed, GlobalAveragePooling1D, Merge, GRU
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
instance_flag=0 #0 for loading data from Local, 1 for FloydHub instance
#MAIN
nb_train_samples = 14280
nb_validation_samples = 6120
lstm_num_timesteps=10 #How many timesteps of features are being fed per sample of a batch. 


batch_size=20 #Samples per batch.
timesteps=10 #Timesteps per sample
features_size=2048


#Define Generator
def generator(features, labels, batch_size, timesteps, flag=0):
	count=0
	print('Generator Active')
	batch_features=np.empty((0,timesteps, features_size))
	batch_labels=np.empty((1,0))

	while True:

		#print (batch_features.shape, batch_labels.shape )
		if flag == 0:
			index= np.random.randint(0, nb_train_samples-1, size = 1, dtype=np.int64)
		else:
			index= np.random.randint(0, nb_validation_samples-1, size = 1, dtype=np.int64)
			
		dataX = features[index,]
		dataY = labels[index]

		#Create One sample of (timesteps, features_size)

		for j in range(1,timesteps):
			
			x = features[index-j,]
			y = labels[index-j,]
			#print(y.shape)
			dataX = np.vstack((dataX,x))
			dataY= np.vstack((dataY,y))

		dataY = np.mean(dataY, axis=0)

		#yield dataX, dataY
		dataX=np.expand_dims(dataX, axis=0)
		dataY=np.expand_dims(dataY, axis=0)
		#print(dataX.shape, dataY.shape)

		batch_features = np.vstack((batch_features,dataX))
		batch_labels = np.column_stack((batch_labels,dataY))

		#print(batch_features.shape, batch_labels.shape)

		count+=1
		if count == batch_size:
			#print("Batch Features Dims , Batch Labels Dims =" batch_features.input_shape, batch_labels.shape)
			#print("Iteration number = ",count)
			count=0
			dataX=[]
			dataY=[]
			TrainY =  np.transpose(batch_labels, (1, 0))
			yield batch_features, TrainY
		
#Build Model
def buildmodel(summary):

	model=Sequential()
	#model.add(Merge([left, right], mode = 'concat'))
	#model.add(Bidirectional(LSTM(128, activation='relu', return_sequences=True), input_shape=(timesteps,features_size)))
	#model.add(Bidirectional(LSTM(128, activation='relu', return_sequences=False)))
	model.add(LSTM(128, activation='relu', recurrent_activation='hard_sigmoid', return_sequences=True, input_shape=(timesteps,features_size)))
	model.add(LSTM(128, activation='relu', return_sequences=False))
	model.add(Dense(128, activation='softplus'))
	model.add(BatchNormalization())
	model.add(Dense(100, activation='softplus'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))
	model.add(Dense(3, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='linear'))


	print('Compiling Model...')
	
	model.compile(optimizer='adam',
		loss='mse',
		metrics=['accuracy'])

	if summary:
		print(model.summary())
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

model = buildmodel(summary=1)
train_generator = generator(X_train, y_train, batch_size, timesteps, 0)
validation_generator = generator(X_train, y_train, batch_size, timesteps, 1)

print('Training...')

model.fit_generator(train_generator, steps_per_epoch=20, epochs=5, verbose=1, validation_data=validation_generator, validation_steps=10)

print('Training Successful - Saving Weights...')

if instance_flag==0:
	model.save_weights('data/lstm_speed_model.h5')
else:
	model.save_weights('/output/lstm_speed_model.h5')
