from keras.models import Sequential
from keras.layers import Dense
from scipy.io import loadmat
import numpy as np

# initialize model
model = Sequential()

# add layers
model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))

# set weights
# weights are Numpy arrays
# somehow set weights from labels.mat

# read matlab file
mat_labels = loadmat('labels.mat', squeeze_me=True, struct_as_record=False)
labels = mat_labels['labels']

model.set_weights(labels)

# configure learning process
model.compile(loss='categorical_crossentropy',
			  optimizer='sgd',
			  metrics=['accuracy'])

# print out summary of model
model.summary()
			  
# train with training data
# training data should be Numpy arrays
#model.fit(x_train, y_train, epochs=5, batch_size=32)
#model.train_on_batch(x_batch, y_batch)

# evaluate performance
#loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)

# generate predictions on new data
#classes = model.predict(x_test, batch_size=128)