from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

data_generator = ImageDataGenerator(
    rescale = 1./255, 
    shear_range = 0.2, 
    zoom_range = 0.2,
    validation_split = 0.25
)

path='/Users/shounakr/Desktop/Final project/Images/CompleteImages/All data (Compressed)'
train = data_generator.flow_from_directory(
    path, 
    target_size = (40, 40), 
    color_mode = 'grayscale',
    batch_size = 32,
    class_mode = 'categorical',
    shuffle = True,
    subset ='training',
    seed = 123
)
valid_set = data_generator.flow_from_directory(
    path, 
    target_size = (40, 40), 
    color_mode = 'grayscale',
    batch_size = 32,
    class_mode = 'categorical',
    shuffle = True,
    subset = 'validation',
    seed = 123
)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape = (40, 40, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dense(16, activation = 'softmax'))
# compile model
adam = tf.keras.optimizers.Adam(lr = 5e-4)
model.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit_generator(train, validation_data = valid_set, epochs = 7, verbose = 1)
model.save("/Users/shounakr/Desktop/model2.h5")