import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def compute_difference(matrix1, matrix2):
    return matrix1 - matrix2

# Specify the file path
file_path = './Datasets/data.csv'

# Read the CSV file
data = pd.read_csv(file_path)


# Convert the pandas DataFrame to a numpy array
data_array = data.to_numpy()

# Remove the last column from the numpy array
data_array = data_array[:, 1:-1]


# Normalize the data for each column independently
data_array_normalized = (data_array - data_array.min(axis=0)) / (data_array.max(axis=0) - data_array.min(axis=0))
#add 1 to all values in columns that are not vessels and creatina to avoid 0
data_array_normalized = data_array_normalized + 1


# Define the autoencoder model
model = Sequential()
model.add(Dense(18, activation='tanh', input_shape=(data_array_normalized.shape[1],)))
model.add(Dense(15, activation='tanh'))
model.add(Dense(10, activation='tanh'))
model.add(Dense(15, activation='tanh'))
model.add(Dense(18, activation='tanh'))
model.add(Dense(data_array_normalized.shape[1], activation='linear'))

#split the data in 20% test and 80% train withouth sklearn
test_size = int(len(data_array_normalized) * 0.2)
train_size = len(data_array_normalized) - test_size
train_data = data_array_normalized[:train_size]
test_data = data_array_normalized[train_size:]
train_copy = train_data.copy()



#define a personalized loss function
#using tensorflow
def custom_loss(y_true, y_pred):
    #get original missing values
    #set last column value of y_true to corresponding value y_pred if y_true is 0
    mask = tf.cast(tf.math.equal(y_true, 0), tf.float32)
    y_true = tf.math.multiply(y_true, mask) + tf.math.multiply(y_pred, 1 - mask)
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Compile and train the model
model.compile(optimizer='adam', loss=custom_loss)
losses = model.fit(train_data, train_copy, validation_data=(test_data, test_data), epochs=20)

#plot
import matplotlib.pyplot as plt
plt.plot(losses.history['loss'])
plt.plot(losses.history['val_loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Training', 'Validation'])
plt.show()

#print loss
print(losses.history['loss'][-1])
print(losses.history['val_loss'][-1])

