import tensorflow as tf
import tensorflowjs as tfjs
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense

IMG_WIDTH = 300
IMG_HEIGHT = 300

def tf_parse(record):
    keys_to_features = {
        "width":     tf.io.FixedLenFeature([], tf.int64, default_value=0),
        "height":     tf.io.FixedLenFeature([], tf.int64, default_value=0),
        "depth":     tf.io.FixedLenFeature([], tf.int64, default_value=0),
        "label":     tf.io.FixedLenFeature([], tf.int64, default_value=0),
        "image_raw": tf.io.FixedLenFeature([], tf.string, default_value='')
    }
    parsed = tf.io.parse_example(record[tf.newaxis], keys_to_features)

    img = tf.image.decode_jpeg(parsed["image_raw"][0], channels=3)   
    img = tf.reshape(img, shape=[parsed["height"][0], parsed["width"][0], parsed["depth"][0]])
    img = tf.image.resize_with_crop_or_pad(img, IMG_HEIGHT, IMG_WIDTH)
    
    label = tf.cast(parsed["label"][0], tf.int64)

    return img, label

# define a function to list tfrecord files.
def list_tfrecord_file(file_list):
    tfrecord_list = []
    for i in range(len(file_list)):
        current_file_abs_path = os.path.abspath(file_list[i])
        if current_file_abs_path.endswith(".tfrecords"):
            tfrecord_list.append(current_file_abs_path)
            print("Found %s successfully!" % file_list[i])
        else:
            pass
    return tfrecord_list

# Traverse current directory
def tfrecord_auto_traversal():
    current_folder_filename_list = os.listdir("./") # Change this PATH to traverse other directories if you want.
    if current_folder_filename_list != None:
        print("%s files were found under current folder. " % len(current_folder_filename_list))
        print("Please be noted that only files end with '*.tfrecords' will be load!")
        tfrecord_list = list_tfrecord_file(current_folder_filename_list)
        if len(tfrecord_list) != 0:
            for list_index in range(len(tfrecord_list)):
                print(tfrecord_list[list_index])
        else:
            print("Cannot find any tfrecords files, please check the path.")
    return tfrecord_list


#dataset = tf.data.TFRecordDataset(filenames = tfrecord_auto_traversal())
dataset = tf.data.TFRecordDataset(filenames = ['petimages2.tfrecords'])
parsed_dataset = dataset.map(tf_parse).shuffle(buffer_size=50000)

val_dataset = parsed_dataset.take(1024).batch(16, drop_remainder=True)
train_dataset = parsed_dataset.skip(1024).batch(16, drop_remainder=True)

model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

#model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3, activation = 'softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_dataset, steps_per_epoch=100, epochs=50,
          validation_data=val_dataset, validation_steps=10)

model.evaluate(val_dataset)

model.save('trained_model.h5')

tfjs.converters.save_keras_model(model, 'tensorjs')