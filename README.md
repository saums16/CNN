# CNN

# Mounting of Drive
from google.colab import drive
drive.mount('/content/drive')

**Preprocessing**

*Importing the Libraries*

# Importing the Libraries
import keras
import tensorflow as tf
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, Conv3D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, RMSprop
from keras.losses import mae, sparse_categorical_crossentropy, binary_crossentropy
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

*Loading the dataset and image resizing*

#Resizing of images to 128*128 and splitting the train and validation set
train_gen=ImageDataGenerator(
                             rescale = 1./255, 
                             validation_split=0.2) # Splitting the validation dataset

train = train_gen.flow_from_directory("/content/drive/My Drive/cloudy_sunny/",
                                            class_mode="binary",
                                            target_size=(128, 128),
                                            color_mode="rgb",
                                            shuffle=True,
                                            batch_size=32,
                                      subset='training')

# Defining the validation set images to 128*128
valid = train_gen.flow_from_directory("/content/drive/My Drive/cloudy_sunny/",
                                            target_size=(128, 128),
                                            color_mode="rgb",
                                            shuffle=True,
                                            batch_size=32,
                                      class_mode='binary',
                                      subset='validation')

# Loading test set and resizing images to 128*128
test_gen=ImageDataGenerator(rescale = 1./255)

test = test_gen.flow_from_directory("/content/drive/My Drive/test/",
                                            class_mode='binary',
                                            target_size=(128, 128),
                                            color_mode="rgb",
                                            shuffle=True,
                                            batch_size=20000)

# Splitting into X_test and Y_test from the test generator
X_test, Y_test = next(test)
print(X_test.shape)
print(Y_test.shape)

print(Y_test.shape)

**Modeling**

# Defining the CNN model
def CNN_model(): 
  
  model = Sequential()
  model.add(Conv2D(filters = 8, kernel_size = 2, padding = 'same', activation = 'relu', input_shape = (128, 128, 3)))
  model.add(Dropout(0.2))
  model.add(MaxPooling2D(pool_size = 2))

  model.add(Conv2D(filters = 16, kernel_size = 2, padding = 'same', activation = 'relu'))
  model.add(Dropout(0.2))
  model.add(MaxPooling2D(pool_size = 2))
 
  model.add(Conv2D(filters = 32, kernel_size = 2, padding = 'same', activation = 'relu'))
  model.add(Dropout(0.2))
  model.add(MaxPooling2D(pool_size = 2))
 
  model.add(Conv2D(filters = 64, kernel_size = 2, padding = 'same', activation = 'relu'))
  model.add(Dropout(0.2))
  model.add(MaxPooling2D(pool_size = 2))
 
  model.add(Conv2D(filters = 128, kernel_size = 2, padding = 'same', activation = 'relu'))
  model.add(Dropout(0.2))
  model.add(MaxPooling2D(pool_size = 2))

 
  model.add(Flatten())
  model.add(Dense(1024, activation='relu'))
  model.add(Dropout(0.2))
  model.add(Dense(1, activation = 'sigmoid'))
 
  model.compile(optimizer=Adam(0.0001), loss=binary_crossentropy, metrics=['accuracy'])
  model.summary()

  return model

# Defining Calculation of time function
import time
class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

*Model Training*

# Model Compilation

model = CNN_model()
 
STEP_SIZE_TRAIN=train.n//train.batch_size
STEP_SIZE_VALID=valid.n//valid.batch_size
 
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=12, restore_best_weights=True, verbose=1)

time_callback = TimeHistory()

with tf.device('/GPU:0'):
# Training of Model
  history = model.fit_generator(train,
                                steps_per_epoch=100,
                                validation_data=valid,
                                validation_steps=100,
                                epochs=32, callbacks= [time_callback])


**Saving the Model**

#Applying model.save_weights() for saving the weights of the model in HDF5 format
model.save_weights("/content/drive/My Drive/thesis/CNN_weights.h5")
model.save("/content/drive/My Drive/thesis/CNN_weights.h5")

# Mounting of Drive
from google.colab import drive
drive.mount('/content/drive')

**Preprocessing**

*Importing the Libraries*

# Importing the Libraries
import keras
import tensorflow as tf
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, Conv3D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, RMSprop
from keras.losses import mae, sparse_categorical_crossentropy, binary_crossentropy
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

*Loading the dataset and image resizing*

#Resizing of images to 128*128 and splitting the train and validation set
train_gen=ImageDataGenerator(
                             rescale = 1./255, 
                             validation_split=0.2) # Splitting the validation dataset

train = train_gen.flow_from_directory("/content/drive/My Drive/thesis/Weather2Class/cloudy_sunny/",
                                            class_mode="binary",
                                            target_size=(128, 128),
                                            color_mode="rgb",
                                            shuffle=True,
                                            batch_size=32,
                                      subset='training')

# Defining the validation set images to 128*128
valid = train_gen.flow_from_directory("/content/drive/My Drive/thesis/Weather2Class/cloudy_sunny/",
                                            target_size=(128, 128),
                                            color_mode="rgb",
                                            shuffle=True,
                                            batch_size=32,
                                      class_mode='binary',
                                      subset='validation')

# Loading test set and resizing images to 128*128
test_gen=ImageDataGenerator(rescale = 1./255)

test = test_gen.flow_from_directory("/content/drive/My Drive/thesis/Weather2Class/test/",
                                            class_mode='binary',
                                            target_size=(128, 128),
                                            color_mode="rgb",
                                            shuffle=True,
                                            batch_size=20000)

# Splitting into X_test and Y_test from the test generator
X_test, Y_test = next(test)
print(X_test.shape)
print(Y_test.shape)

print(Y_test.shape)

**Modeling**

# Defining the CNN model
def CNN_model(): 
  
  model = Sequential()
  model.add(Conv2D(filters = 8, kernel_size = 2, padding = 'same', activation = 'relu', input_shape = (128, 128, 3)))
  model.add(Dropout(0.2))
  model.add(MaxPooling2D(pool_size = 2))

  model.add(Conv2D(filters = 16, kernel_size = 2, padding = 'same', activation = 'relu'))
  model.add(Dropout(0.2))
  model.add(MaxPooling2D(pool_size = 2))
 
  model.add(Conv2D(filters = 32, kernel_size = 2, padding = 'same', activation = 'relu'))
  model.add(Dropout(0.2))
  model.add(MaxPooling2D(pool_size = 2))
 
  model.add(Conv2D(filters = 64, kernel_size = 2, padding = 'same', activation = 'relu'))
  model.add(Dropout(0.2))
  model.add(MaxPooling2D(pool_size = 2))
 
  model.add(Conv2D(filters = 128, kernel_size = 2, padding = 'same', activation = 'relu'))
  model.add(Dropout(0.2))
  model.add(MaxPooling2D(pool_size = 2))

 
  model.add(Flatten())
  model.add(Dense(1024, activation='relu'))
  model.add(Dropout(0.2))
  model.add(Dense(1, activation = 'sigmoid'))
 
  model.compile(optimizer=Adam(0.0001), loss=binary_crossentropy, metrics=['accuracy'])
  model.summary()

  return model


*Model Training*

# Model Compilation

model = CNN_model()
 
STEP_SIZE_TRAIN=train.n//train.batch_size
STEP_SIZE_VALID=valid.n//valid.batch_size
 
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=12, restore_best_weights=True, verbose=1)

time_callback = TimeHistory()

with tf.device('/GPU:0'):
# Training of Model
  history = model.fit_generator(train,
                                steps_per_epoch=100,
                                validation_data=valid,
                                validation_steps=100,
                                epochs=32, callbacks= [time_callback])


**Saving the Model**

#Applying model.save_weights() for saving the weights of the model in HDF5 format
model.save_weights("/content/drive/My Drive/CNN_weights.h5")
model.save("/content/drive/My Drive/CNN_weights.h5")
