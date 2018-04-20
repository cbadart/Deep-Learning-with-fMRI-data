import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from scipy.io import loadmat
import keras
import numpy as np
import os, glob
import pprint

def load_data():
	data = np.zeros(shape=(100, 1940, 116))
	i = 0
	# get all the data from the matlab files
	for root, dirnames, filenames in os.walk('fMRI'):
		for file in filenames:
			file_path = os.path.join(root, file)
			#print(file_path)
			mat = loadmat(file_path)
			data[i] = mat['data']
			i = i + 1
	print(data.shape)
	return data

def ten_fold_xv(data):
	# does 10-fold cross validation
	results = dict()
	for i in range(10):
		test = np.zeros(shape=(10, 1940, 116))
		train = np.zeros(shape=(90, 1940, 116))
		# partition data into test and train
		test_indices = range(i*10, i*10+10)
		test_i = 0
		train_i = 0
		for d in range(100):
			if d in test_indices:
				test[test_i] = data[d]
				test_i = test_i + 1
			else:
				train[train_i] = data[d]
				train_i = train_i + 1
		
		# train model
		
		# evaluate performance
		#loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
		#results[i] = loss_and_metrics["accuracy"] #??? maybe ???
		
	# calculate overall accuracy
	print("----- RESULTS -----")
	total = 0
	for instance in results:
		total = total + results[instance]
		print("{}: {}".format(instance, results[instance]))
	avg = total / 10.0
	print("\nOverall: {}".format(avg))
	
def main():
	# initialize model
	model = Sequential()

	# add layers
	# input layer: data is 1940 x 116 so input layer should have 116 nodes

	model.add(Dense(units=116, activation='relu', input_shape=(1940,116,)))
	model.add(Flatten())
	# hidden layer: ~2/3(n_in) + n_out = 84
	model.add(Dense(units=84, activation='relu'))

	# output layer: one node per class label when using softmax, 7 classes so 7 nodes
	model.add(Dense(units=8, activation='softmax'))

	# read matlab label file
	mat_labels = loadmat('labels.mat', squeeze_me=True, struct_as_record=False)
	labels = mat_labels['labels']

	# configure learning process
	model.compile(loss='categorical_crossentropy',
				  optimizer='sgd',
				  metrics=['accuracy'])

	# print out summary of model
	model.summary()
	
	# load fMRI data from the matlab files

	data  = load_data()
	pprint.pformat(data)
	train = data[0:89]
	print("train shape: {}".format(train.shape))
	test = data[90:99]
	print("test shape: {}".format(test.shape))
			
	#model.fit(x_train, y_train, epochs=5, batch_size=32)
	model.train_on_batch(train, keras.utils.to_categorical(labels))
	#ten_fold_xv(data)
	# evaluate performance
	loss_and_metrics = model.evaluate(test, labels, batch_size=128)
	print(loss_and_metrics)

	# generate predictions on new data
	#classes = model.predict(x_test, batch_size=128)

if __name__ == '__main__':
	main()