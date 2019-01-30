import cv2
import numpy as np
import matplotlib.pyplot as  plt
from sklearn import model_selection
import os
import random
from sklearn.utils import shuffle
from keras.utils.np_utils import to_categorical
from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten, Lambda, Activation, MaxPooling2D, Reshape, Input, concatenate
from keras.optimizers import Adam
from keras.models import load_model

IMAGE_HEIGHT = 320
IMAGE_WIDTH = 160
IMAGE_CHANNEL = 3
BATCH_SIZE  = 112
NUM_CLASSES =4
EPOCHS = 2
def get_files(filename):
    class_train = []
    label_train = []
    for train_class in os.listdir(filename):
        for pic in os.listdir(filename+train_class):
            class_train.append(filename+train_class+'/'+pic)

            if (train_class == 'red'):
                   # data[1] = 0
                   label = 0
            if (train_class == 'green'):
                   # data[1] = 2
                   label = 2
            if (train_class == "yellow"):
                   # data[1] = 1
                   label = 1
            if (train_class == 'none'):
                   # data[1] = 4
                   label = 3

            label_train.append(label)

    return class_train,label_train

filename = "sim_imgs/"

image_list,y_labe = get_files(filename)

print(len(y_labe))
print (y_labe)
print(len(image_list))
print (image_list[2])


def random_brightness(image):
    # Convert 2 HSV colorspace from RGB colorspace
    #img = cv2.imread(image)
    #standard_im = cv2.resize(image, (320, 160))
    stand_img = cv2.resize(image, (160,320), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(stand_img, cv2.COLOR_BGR2HSV)
    # Generate new random brightness
    rand = random.uniform(0.3, 1.0)
    hsv[:, :, 2] = rand*hsv[:, :, 2]
    # Convert back to RGB colorspace
    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return new_img

#test
index = random.randint(0,len(image_list))
image = cv2.imread(image_list[index])
test_img = random_brightness(image)
plt.imshow(test_img)
plt.show()
#model test
pre_img =cv2.resize(image, (160, 320), interpolation=cv2.INTER_AREA)
model = load_model("model-1.30.h5")

pre_result = model.predict_classes(np.asarray([pre_img]))
predict_path = image_list[index]
print ("perdict:", predict_path,pre_result )

# loads image
def get_image(index, data, labels):
    # pair image and color clasiffication
    image = cv2.imread(index)
    color = labels[data.index(index)]

    return [image, color]

#generator train data
def generator(data, labels, has_augment = False):
    while True:
        # Randomize the indices to make an array
        # indices_arr = np.random.permutation(data.count()[0])
        data, labels = shuffle(data, labels)
        for batch in range(0, len(data), BATCH_SIZE):
            # slice out the current batch according to batch-size
            current_batch = data[batch:(batch + BATCH_SIZE)]
            #print(current_batch)
            # initializing the arrays, x_train and y_train
            x_train = np.empty(
                [0, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL], dtype=np.float32)
            y_train = np.empty([0], dtype=np.int32)

            for i in current_batch:
                [image, color] = get_image(i, data, labels)
                nor_image = random_brightness(image)
                x_train = np.append(x_train, [nor_image], axis=0)
                y_train = np.append(y_train, [color])
            y_train = to_categorical(y_train, num_classes=NUM_CLASSES)

            yield (x_train, y_train)

def get_model():
    model = Sequential()
    input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL)
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(input_shape)))
    model.add(Conv2D(24, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(36, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(48, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dropout(.5))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dropout(.75))
    model.add(Dense(10))
    model.add(Dense(4))
    #model.summary()
    # plot_model(model, to_file='model.png')
    model.add(Activation("softmax"))
    #model.add(Lambda(lambda x: K.tf.nn.softmax(x)))
    model.compile(optimizer=Adam(lr=5e-4),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    return model


if __name__ == "__main__":

    #data_num, data, labels = create_labeled_list()
    data,labels = shuffle(image_list, y_labe)
    #data,labels = (image_list, y_labe)
    # Split data into random training and validation sets
    d_train,  d_valid, l_train, l_valid = model_selection.train_test_split(data, labels, test_size=0.2)

    train_gen = generator(d_train, l_train,True)
    validation_gen = generator(d_valid, l_valid,False)

    model = get_model()

    # checkpoint to save best weights after each epoch based on the improvement in val_loss
    #checkpoint = ModelCheckpoint(MODEL_FILE_NAME, monitor='val_loss', verbose=1,save_best_only=True, mode='min',save_weights_only=False)
    #callbacks_list = [checkpoint] #,callback_each_epoch]
    #model.compile(optimizer=Adam(0.0001), loss="mse")

    print('Training started....')

    history = model.fit_generator(
        train_gen,
        steps_per_epoch=len(d_train),
        epochs=EPOCHS,
        validation_data=validation_gen,
        validation_steps=len(d_valid),
        verbose=1
    )
    #save
    model.save('model.h5')  # creates a HDF5 file 'model.h5'

    ### plot the training and validation loss for each epoch
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

    #model.save('model.h5')  # creates a HDF5 file 'model.h5'


