import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

#to be sure we use GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"

#to use only required memory on gpu, and not use all memory
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

#AI hyperparameters
batch_size = 256
epochs = 10

#dataset loading
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

input_img = Input(shape=(28,28,1))

#generate noisy images
noise_factor = 0.6
x_train_noisy = x_train + noise_factor * np.random.normal(size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(size=x_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

#network definition
x = Conv2D(64, (3,3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(64, (3,3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2,2))(x)
x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = UpSampling2D((2,2))(x)
decoded = Conv2D(1, (3,3), activation='sigmoid', padding='same')(x)
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

#network learning
autoencoder.fit(x_train_noisy,
				x_train,
				epochs=epochs,
				batch_size=batch_size,
				shuffle=True,
				validation_data=(x_test_noisy, x_test))

#use if you want to generate encoded images
#encoder = Model(input_img, encoded)
#encoded_imgs = encoder.predict(x_test_noisy)

#denoised images will be obtained by using the autoencoder
denoised_imgs = autoencoder.predict(x_test_noisy)

#number of images displayed
n = 20

#index of the first displayed image
start = 200

#display noisy images and their denoised result obtained by the AI
plt.figure(figsize=(20,4))
for i in range(n):
	ax = plt.subplot(2,n,i+1)
	plt.imshow(x_test_noisy[i+start].reshape(28,28))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
	ax = plt.subplot(2,n,i+1+n)
	plt.imshow(denoised_imgs[i+start].reshape(28,28))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
plt.show()