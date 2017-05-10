import os
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.optimizers import SGD, RMSprop

from keras.layers import Dropout, Flatten, Dense, Convolution2D, LSTM, Bidirectional, Activation, TimeDistributed, GlobalAveragePooling1D, Merge, GRU
from keras import applications
from keras.layers.normalization import BatchNormalization
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import random_projection
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
# dimensions of our images.
#img_width, img_height = 224, 224

# fix random seed for reproducibility
np.random.seed(10)
#train_data_dir = '/input/data/images/train'
#validation_data_dir = '/input/data/images/validation'
nb_train_samples = 14280
nb_validation_samples = 6120
instance_flag=0 #0 for loading data from Local, 1 for FloydHub instance
#MAIN
nb_train_samples = 14280
nb_validation_samples = 6120
#lstm_num_timesteps=10 #How many timesteps of features are being fed per sample of a batch. 


batch_size=32 #Samples per batch.
timesteps=3 #Timesteps per sample
#features_size=2048

global start
start=1
#Define Generator
def generator(features, labels, batch_size, timesteps, start, flag=0):
	count=0
	print('Generator Active')
	batch_features=np.empty((0,timesteps, features_size))
	batch_labels=np.empty((1,0))
	
	while True:

		#print (batch_features.shape, batch_labels.shape )

		if flag == 0 :
			#index= np.random.randint(0, nb_train_samples-1, size = 1, dtype=np.int64)
			
			if start==1:
			 	index= np.random.randint(0, nb_train_samples-10, size = 1, dtype=np.int64)
			 	start=3
			else: 
				index = old_index+1
		
		else :

			if start==1:
			 	index = timesteps
			 	start=3
			else:
				index = old_index+1

			#index= np.random.randint(0, nb_validation_samples-1, size = 1, dtype=np.int64)


		dataX = features[index,]
		dataY = labels[index]
		old_index=index
		#print('Currently Processing this frame and 6 frames behind', index)

		#Create One sample of (timesteps, features_size)

		for j in range(1,timesteps):
			pointer = index-j
			#print(pointer)
			x = features[pointer,]
			y = labels[pointer,]
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
		#print count
		if count == batch_size:
			#print("Batch Features Dims , Batch Labels Dims =" batch_features.input_shape, batch_labels.shape)
			#print("Iteration number = ",count)
			count=0
			dataX=[]
			dataY=[]
			TrainY =  np.transpose(batch_labels, (1, 0))
			#print(batch_features.shape, TrainY.shape)
			#print('A batch is being yielded at index and ', index)
			yield batch_features, TrainY
		
#Build Model
def buildmodel(summary):
	#Define hyperparameters and compile model"""
    
	lr = 0.001
	weight_init='glorot_uniform'
	loss = 'mean_absolute_error'

	model=Sequential()
	#model.add(Merge([left, right], mode = 'concat'))
	#model.add(Bidirectional(LSTM(128, activation='relu', return_sequences=True), input_shape=(timesteps,features_size)))
	#model.add(Bidirectional(LSTM(128, activation='relu', return_sequences=False)))
	model.add(BatchNormalization(input_shape=(timesteps, features_size)))
	model.add(Bidirectional(GRU(100, activation='relu', kernel_initializer=weight_init, recurrent_activation='hard_sigmoid' , return_sequences=True)))
	#model.add(BatchNormalization())
	model.add(Bidirectional(GRU(100, activation='relu', kernel_initializer=weight_init, recurrent_activation='hard_sigmoid',return_sequences=False)))
	model.add(Dense(100, init=weight_init, activation='relu'))
	model.add(Dropout(0.8))
	model.add(Dense(50, init=weight_init, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, init=weight_init, activation='linear'))


	print('Compiling Model...')
	#sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(optimizer=RMSprop(lr),
		loss=loss,
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
	#labels.astype(float)
else:
	train_data = np.load(open('/input/bottleneck_features_train.npy'))
	validation_data = np.load(open('/input/bottleneck_features_validation.npy'))
	labels = np.loadtxt('/input/train.txt')
	#labels.astype(float)


scaler = MinMaxScaler(feature_range=(0, 1))
speeds = scaler.fit_transform(labels)
#speeds.astype(float)
print speeds
y_train= speeds[0:nb_train_samples]
y_validation= speeds[nb_train_samples:len(labels)]
#print keras.backend.shape(train_data)
#print keras.backend.shape(validation_data)

x_train=np.reshape(train_data, (nb_train_samples, -1))
x_validation=np.reshape(validation_data, (nb_validation_samples, -1))

transformer = random_projection.GaussianRandomProjection(eps=0.5)
X_train= transformer.fit_transform(x_train)

features_size=X_train.shape[1]

transformer = random_projection.SparseRandomProjection(features_size)
X_validation=transformer.fit_transform(x_validation)

X_train.astype(float)
X_validation.astype(float)

print('Final Data Shape = [X_train, X_validation, y_train, y_validation]', X_train.shape , X_validation.shape, y_train.shape, y_validation.shape)

print('Building Model...')

model = buildmodel(summary=1)
train_generator = generator(X_train, y_train, batch_size, timesteps, 1, 0)
validation_generator = generator(X_validation, y_validation, batch_size, timesteps, 1, 1)


print('Training...')

earlyStopping= EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
                       write_graph=True, write_images=False)


if instance_flag==0:
	checkpointer = ModelCheckpoint(filepath="./dashcam_weights.hdf5", verbose=1, save_best_only=True)
else:
	checkpointer = ModelCheckpoint(filepath="/output/dashcam_weights.hdf5", verbose=1, save_best_only=True)

training = model.fit_generator(train_generator, steps_per_epoch=10, epochs=10, verbose=1, validation_data=validation_generator, validation_steps=5, callbacks=[tensorboard, earlyStopping, checkpointer])

print('Training Successful - Saving Weights...')

loss_history = training.history["val_loss"]
accuracy_history = training.history["val_acc"]

numpy_loss_history = np.array(loss_history)
numpy_accuracy_history = np.array(accuracy_history)
np.savetxt("./loss_history.txt", numpy_loss_history, delimiter=",")
np.savetxt("./accuracy_history.txt", numpy_accuracy_history, delimiter=",")

#print('Evaluating on Validation Dataset')
#evaluate_generator(self, validation_generator, steps=nb_validation_samples, max_q_size=10, workers=1, pickle_safe=False)

#if instance_flag==0:
#	model.save_weights('data/dashcam_weights.hdf5')
#else:
#	model.save_weights('/output/lstm_speed_model_weights.hdf5')
