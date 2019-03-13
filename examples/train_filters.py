from __future__ import print_function

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, Flatten, Conv2D, Permute, Input, MaxPooling2D, Dropout, Concatenate
import keras.backend as K
from datetime import datetime
import os

# directory for saving the model
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = datetime.now().strftime("ILSVRC-CNN3.h5")

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 128

img_size = (100, 120)

train_generator = train_datagen.flow_from_directory(
        '/imagenet_mini/train',
        target_size=img_size,
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    '/imagenet_mini/val',
    target_size=img_size,
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical')


input_shape = img_size + (1,)
input_tensor = Input(shape=input_shape)
input_permuted = Permute((1, 2, 3))(input_tensor)
t = Conv2D(32, (7, 7), strides=(4, 4), activation='relu', name="conv2D_1")(input_permuted)
t = Conv2D(32, (5, 5), strides=(2, 2), activation='relu', name="conv2D_2")(t)
t = Flatten()(t)
out = Dense(6, activation='softmax')(t)  # number of classes
model = Model(inputs=input_tensor, outputs=out)
print(model.summary())

# initiate optimizer
opt = keras.optimizers.Adam()

# Let's train the model
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.fit_generator(
        train_generator,
        steps_per_epoch=73439/batch_size,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=18374/batch_size)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)



