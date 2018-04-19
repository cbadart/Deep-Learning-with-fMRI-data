from keras.models import Sequential
from keras.layers import Dense
from scipy.io import loadmat
import numpy as np
import os, glob

# initialize model
model = Sequential()

# add layers
# input layer: data is 1940 x 116 so input layer should have 116 nodes
model.add(Dense(units=116, activation='relu', input_dim=116))
# hidden layer: ~2/3(n_in) + n_out = 84
model.add(Dense(units=84, activation='relu'))
# output layer: one node per class label when using softmax, 7 classes so 7 nodes
model.add(Dense(units=7, activation='softmax'))

# read matlab label file
mat_labels = loadmat('labels.mat', squeeze_me=True, struct_as_record=False)
labels = mat_labels['labels']

# configure learning process
model.compile(loss='categorical_crossentropy',
			  optimizer='sgd',
			  metrics=['accuracy'])

# print out summary of model
model.summary()

traing_data = []

# train with training data
# training data should be Numpy arrays
#for root, dirnames, filenames in os.walk('fMRI'):
#	for file in filenames:
#		train = laodmat(file)
#		traing_data.append(train['data'])

train1subj = loadmat('fMRI/subj001.mat')['data']
print(train1subj.shape)
		
#model.fit(x_train, y_train, epochs=5, batch_size=32)
model.train_on_batch(train1subj, labels)

# evaluate performance
#loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)

# generate predictions on new data
#classes = model.predict(x_test, batch_size=128)