#Importing libraries
from google.colab import drive
drive.mount('/content/drive')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# for handling imbalancing
from imblearn.under_sampling import NearMiss
from keras.utils import np_utils

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report , confusion_matrix

import keras
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization

from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import SGD
import warnings
warnings.filterwarnings('ignore')
# Data Exploring
df = pd.read_csv('/content/drive/MyDrive/CI/A_Z Handwritten Data.csv')
df.shape
df.tail()
# Data preprocessing
# getting target variable
y = df['0']
del df['0']
# Dealing with imbalanced target
x = y.replace([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25], ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])
x
nM = NearMiss()
X_data, y_data = nM.fit_resample(df, y)
# Encoding
#One-Hot-Encoding of the target.
y = np_utils.to_categorical(y_data)
# Define the classification of 26 alphabets.
num_classes = y.shape[1]
num_classes
y.shape , X_data.shape
# Normalization
# preprocessing technique used in machine learning to scale the feature values 
# to a specific range, typically between 0 and 1. 

X_data = X_data / 255
X_data
# Visualization
X_data = np.array(X_data)
# The reshape() method returns a new reshaped numpy array, and does 
# not modify the original array X_data
X_data = X_data.reshape(-1,28,28,1) 


# Showing few images

f, ax = plt.subplots(2,5)
f.set_size_inches(10,10)
k = 0
for i in range(2):
    for j in range(5):
      #function call that displays a 2D grayscale image of a digit from the MNIST dataset.
        ax[i,j].imshow(X_data[k].reshape(28,28), cmap='gray')
        k += 1
    plt.tight_layout()


# Train test split
#train_test_split( ) basically it splits the dataset into training and testing.
# 80 % to training and 20% to testing

X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=0.2 ,random_state=102)

# Model 1
#Build an ordinary "Deep Learning" model with CNN and maxpooling by using Keras.
model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
#Choose an optimizer and compile the model.
model.compile(optimizer = Adam(learning_rate = 0.01), loss = 'categorical_crossentropy', metrics = ['accuracy'])
#And print the summary of the model.
print(model.summary())

history = model.fit(X_train,y_train,epochs=15, batch_size=128, validation_data=(X_test,y_test))

#  Save the model

model.save('/content/drive/MyDrive/CI/model1.h5')

#  Plotting the graph of loss and accuracy
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('Loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()

# Model 2
model2 = Sequential()

model2.add(Conv2D(64, (5, 5), input_shape=(28, 28, 1), activation='relu', padding="same"))
model2.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu',padding="same"))
model2.add(MaxPooling2D(pool_size=(2, 2)))

model2.add(Conv2D(128, (3, 3), activation='relu', padding="same"))
model2.add(Conv2D(128, (3, 3), activation='relu', padding="same"))
model2.add(MaxPooling2D(pool_size=(2, 2)))

model2.add(Dropout(0.2))

model2.add(Flatten())
model2.add(Dense(128, activation='relu'))
model2.add(Dense(num_classes, activation='softmax'))

model2.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
print(model2.summary())

history = model2.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=256, verbose=2)

model2.save('/content/drive/MyDrive/CI/model2.h5')

# Final evaluation of the model
scores = model2.evaluate(X_test,y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('Loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2

# Load the Keras model
model = keras.models.load_model('/content/drive/MyDrive/CI/model1.h5')

# Load the image
image = cv2.imread('/content/drive/MyDrive/CI/A.jpeg', 0)
image = cv2.resize(image, (28, 28)) 

# Preprocess the image
image = np.expand_dims(image, axis=0)  
image = image / 255.0  

# Make predictions on the image
predictions = model.predict(image)

# Print the predicted class
predicted_class = np.argmax(predictions)
print('Predicted class:', predicted_class)


# Model3
model3 = Sequential()

model3.add(Conv2D(64, (5, 5), input_shape=(28, 28, 1), activation='relu', padding="same"))
model3.add(Conv2D(64, (5, 5), input_shape=(28, 28, 1), activation='relu',padding="same"))
model3.add(MaxPooling2D(pool_size=(2, 2)))

model3.add(Conv2D(128, (3, 3), activation='relu', padding="same"))
model3.add(Conv2D(128, (3, 3), activation='relu', padding="same"))
model3.add(MaxPooling2D(pool_size=(2, 2)))

model3.add(Dropout(0.2))

model3.add(Flatten())
model3.add(Dense(128, activation='relu'))
model3.add(Dense(num_classes, activation='softmax'))

model3.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
print(model3.summary())

history = model3.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=256, verbose=2)
model3.save('/content/drive/MyDrive/CI/model3.h5')

# Final evaluation of the model
scores = model3.evaluate(X_test,y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))


plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('Loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()

# Image Prediction


import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2

def predict_image(filename):
  # Load the Keras model
  model = keras.models.load_model('/content/drive/MyDrive/CI/model1.h5')

  # Load the image
  image = cv2.imread('/content/drive/MyDrive/CI/'+filename, 0)
  image = cv2.resize(image, (28, 28))  # Resize the image to the input shape of the model

  # Preprocess the image
  image = np.expand_dims(image, axis=0)  # Add batch dimension
  image = image / 255.0  # Normalize the pixel values to be between 0 and 1

  # Make predictions on the image
  predictions = model.predict(image)

  # Print the predicted class
  predicted_class = np.argmax(predictions)
  print('Predicted class:', predicted_class)
  print("The Alphabet Predicted is "+chr(ord('A')+predicted_class))

  img = plt.imread('/content/drive/MyDrive/CI/'+filename)
  plt.imshow(img)

predict_image('A.jpeg')






