import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D

# CNN Model
def def_model(in_shape, nb_classes):
    
    model = Sequential()

    model.add(Conv2D(64, (3, 3), activation='relu', name='block1_conv', input_shape=in_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block1_pool'))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation='relu', name='block2_conv'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block2_pool'))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), activation='relu', name='block3_conv'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block3_pool'))
    model.add(Dropout(0.25))

    model.add(Conv2D(512, (3, 3), activation='relu', name='block4_conv'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block4_pool'))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu', name='fc1'))
    model.add(Dropout(0.5, name='dropout1'))
    model.add(Dense(4096, activation='relu', name='fc2'))
    model.add(Dropout(0.5, name='dropout2'))

    model.add(Dense(nb_classes, activation='softmax', name='predictions'))
    
    return model
