import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import  MaxPooling2D
from keras.layers import Flatten, Dense


from keras_preprocessing import image

img_width, img_height = 64, 64

def create_model():
    model = Sequential()


    model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))


    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Flatten())


    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    model.summary()
    return model


from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

training_set = train_datagen.flow_from_directory(r'C:\Users\User\Desktop\Class 3.2\Extra\new Code\dataset\training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                            class_mode='binary')

########################
model = create_model()
model.load_weights('./currnencymodel.h5')
test_image = image.load_img(r'C:\Users\User\Desktop\Class 3.2\Extra\new Code\dataset\Single prediction\fake1.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = model.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'Real'
else:
    prediction = 'Fake'
print("ans: ",prediction)