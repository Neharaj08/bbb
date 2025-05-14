Basic Tensor Operations (Tensor Creation, Arithmetic, Reshaping, etc.)
python
Copy
Edit
import tensorflow as tf
import numpy as np

# Tensor creation
tensor = tf.constant([100, 200, 300])
print("Tensor Shape:", tensor.shape)
print("Data Type:", tensor.dtype)

# Element-wise addition
ts1 = tf.constant(np.random.rand(2, 3))
ts2 = tf.constant(np.random.rand(2, 3))
result_add = tf.add(ts1, ts2)

# Element-wise subtraction
a = tf.constant([10, 20, 30], dtype=tf.float32)
b = tf.constant([5, 15, 25], dtype=tf.float32)
result_sub = tf.subtract(a, b)

# Element-wise multiplication
result_mul = tf.multiply(ts1, ts2)

# Element-wise division (with zero division check)
tensor1 = tf.constant([6, 8, 12, 15], dtype=tf.float32)
tensor2 = tf.constant([2, 3, 4, 0], dtype=tf.float32)
result_div = tf.where(tensor2 != 0, tf.divide(tensor1, tensor2), tf.zeros_like(tensor1))

# Reshaping tensor
initial_tensor = tf.constant([1, 2, 3, 4])
reshaped_tensor = tf.reshape(initial_tensor, (2, 2))

# Squaring tensor
a = tf.constant([-5, -7, 2, 5, 7], dtype=tf.float64)
res_square = tf.math.square(a)

# Broadcasting
tensor_b = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
broadcast_result = tensor_b + 5

# Concatenating tensors
tensor_a = tf.constant([[1, 2], [3, 4]])
tensor_b = tf.constant([[5, 6], [7, 8]])
concat_tensor = tf.concat([tensor_a, tensor_b], axis=0)
-------------------

2. Simple Sequential CNN (MNIST)
python
Copy
Edit
import tensorflow as tf
import numpy as np

from tensorflow.keras.datasets import mnist

(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
X_train_full = X_train_full / 255.0
X_test = X_test / 255.0

X_train, X_val = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_val = y_train_full[:-5000], y_train_full[-5000:]

X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]
X_test = X_test[..., np.newaxis]

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="relu", kernel_initializer="he_normal"),
    tf.keras.layers.Conv2D(64, kernel_size=3, padding="same", activation="relu", kernel_initializer="he_normal"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(128, activation="relu", kernel_initializer="he_normal"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
model.save("./models/my_mnist_cnn_model.keras")
model.evaluate(X_test, y_test)
------
3. Fine-tuning ResNet50
python
Copy
Edit
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os

data_dir = "path/to/dataset"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")

img_size = (224, 224)
batch_size = 32
num_classes = len(os.listdir(train_dir))
epochs = 10

datagen_train = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

datagen_val = ImageDataGenerator(rescale=1.0/255)

train_generator = datagen_train.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical')
val_generator = datagen_val.flow_from_directory(val_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical')

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
out = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=out)
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_generator, validation_data=val_generator, epochs=epochs)

-------

4. Optimizer Comparison (Adam vs RMSprop)
python
Copy
Edit
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, RMSprop
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv('Data/winequality-red.csv', sep=';')
X = data.drop(['quality'], axis=1)
y = data['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

def create_model():
    model = Sequential()
    model.add(Dense(100, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='linear'))
    return model

model = create_model()
model.compile(optimizer=Adam(), loss='mse', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=128)
-------
Laboratory Task 7: Implement a basic RNN for sequence prediction.
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
# Generate synthetic sequential data (e.g., a sine wave)
def generate_sequence(n_timesteps):
 x = np.linspace(0, 50, n_timesteps)
 y = np.sin(x)
 return y
# Prepare dataset
n_timesteps = 100
sequence = generate_sequence(n_timesteps)
# Create input-output pairs for training (Sliding window method)
X, y = [], []
seq_length = 10 # Number of previous steps used for prediction
for i in range(len(sequence) - seq_length):
 X.append(sequence[i:i+seq_length])
 y.append(sequence[i+seq_length])
X, y = np.array(X), np.array(y)
# Reshape input for RNN [samples, timesteps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))
# Build RNN model
model = Sequential([
 SimpleRNN(10, activation='relu', return_sequences=False, input_shape=(seq_length, 1)),
 Dense(1)
])
model.compile(optimizer='adam', loss='mse')
# Train model
model.fit(X, y, epochs=100, verbose=1)
# Make predictions
predictions = model.predict(X)
# Print expected vs. predicted output
print(f"Expected Output: {y[:5]}")
print(f"Predicted Output: {predictions[:5].flatten()}")
OUTPUT:
The model will try to learn the sine wave pattern and predict future values.
The Mean Squared Error (MSE) loss should gradually decrease.
The printed predicted values should be close to the expected sine wave values.
--------------
8: Build an LSTM-based model for time-series forecasting or
text generation
import tensorflow as tf
import numpy as np
import string
# Load text data
text = open("shakespeare.txt", "r").read().lower()
chars = sorted(set(text))
# Map characters to indices
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}
# Convert text to sequence of numbers
seq_length = 100
sequences = []
next_chars = []
for i in range(len(text) - seq_length):
 sequences.append([char_to_idx[c] for c in text[i:i+seq_length]])
 next_chars.append(char_to_idx[text[i+seq_length]])
 X = np.array(sequences)
y = np.array(next_chars)
# Reshape input for LSTM
X = X.reshape(X.shape[0], X.shape[1], 1) / len(chars)
# Build LSTM model
model = tf.keras.Sequential([
 tf.keras.layers.LSTM(128, input_shape=(seq_length, 1), return_sequences=True),
 tf.keras.layers.LSTM(128),
 tf.keras.layers.Dense(len(chars), activation="softmax")
])
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
# Train the model
model.fit(X, y, epochs=20, batch_size=64)
# Function to generate text
def generate_text(seed_text, length=200):
 generated = seed_text
 for _ in range(length):
 x_input = np.array([[char_to_idx[c] for c in generated[-seq_length:]]]) / len(chars)
 x_input = x_input.reshape(1, seq_length, 1)
 predicted_idx = np.argmax(model.predict(x_input))
 generated += idx_to_char[predicted_idx]
 return generated
# Generate new text
print(generate_text("shall i compare thee to a summer's day? "))
OUTPUT:
A trained LSTM model that generates text similar to the dataset.
shall i compare thee to a summer's day? thou art more lovely and more temperate:
rough winds do shake the darling buds of may,
and summer's lease hath all too short a date...

-------------
6 
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
Code # Alternatively, you can use the same dataset made readily available
by keras Using the following lines of code:
(X_train, y_train), (X_test, y_test) =
tf.keras.datasets.fashion_mnist.load_data()
Code plt.imshow(X_train[0], cmap="gray")
X_train.shape
o/p (60000, 28, 28)
Code X_test.shape
o/p (10000, 28, 28)
STEP #2: PERFORM DATA VISUALIZATION
Code # Let's view some images!
i = random.randint(1,60000) # select any random index from 1 to
60,000
plt.imshow( X_train[i] , cmap = 'gray') # reshape and plot the image
label = y_train[i]
label
o/p 4
Code # Let's view more images in a grid format
# Define the dimensions of the plot grid
W_grid = 15
L_grid = 15
# fig, axes = plt.subplots(L_grid, W_grid)
# subplot return the figure object and axes object
# we can use the axes object to plot specific figures at various
locations
fig, axes = plt.subplots(L_grid, W_grid, figsize = (17,17))
axes = axes.ravel() # flaten the 15 x 15 matrix into 225 array
n_training = len(X_train) # get the length of the training dataset
# Select a random number from 0 to n_training
for i in np.arange(0, W_grid * L_grid): # create evenly spaces
variables
 # Select a random number
 index = np.random.randint(0, n_training)
 # read and display an image with the selected index
 axes[i].imshow( X_train[index] )
 axes[i].set_title(y_train[index], fontsize = 8)
 axes[i].axis('off')
plt.subplots_adjust(hspace=0.4)
STEP #3: PERFORM DATA PREPROCESSING
Code X_train = X_train / 255
X_test = X_test / 255
Code noise_factor = 0.3
noise_dataset = []
for img in X_train:
 noisy_image = img + noise_factor * np.random.randn(*img.shape)
 noisy_image = np.clip(noisy_image, 0., 1.)
 noise_dataset.append(noisy_image)
Code noise_dataset = np.array(noise_dataset)
Code noise_dataset.shape
o/p (60000, 28, 28)
Code plt.imshow(noise_dataset[22], cmap="gray")
noise_test_set = []
for img in X_test:
 noisy_image = img + noise_factor * np.random.randn(*img.shape)
Department of CSE, DSCE Deep Learning Lab - VI semester
 noisy_image = np.clip(noisy_image, 0., 1.)
 noise_test_set.append(noisy_image)

noise_test_set = np.array(noise_test_set)
noise_test_set.shape
o/p (10000, 28, 28)
STEP #4: BUILD AND TRAIN AUTOENCODER
DEEP LEARNING MODEL
Code autoencoder = tf.keras.models.Sequential()
#Encoder
autoencoder.add(tf.keras.layers.Conv2D(filters=16, kernel_size=3,
strides=2, padding="same", input_shape=(28, 28, 1)))
autoencoder.add(tf.keras.layers.Conv2D(filters=8, kernel_size=3,
strides=2, padding="same"))
#Encoded image
autoencoder.add(tf.keras.layers.Conv2D(filters=8, kernel_size=3,
strides=1, padding="same"))
#Decoder
autoencoder.add(tf.keras.layers.Conv2DTranspose(filters=16,
kernel_size=3, strides=2, padding="same"))
autoencoder.add(tf.keras.layers.Conv2DTranspose(filters=1,
kernel_size=3, strides=2, activation='sigmoid', padding="same"))
Code autoencoder.compile(loss='binary_crossentropy',
optimizer=tf.keras.optimizers.Adam(lr=0.001))
autoencoder.summary()
autoencoder.fit(noise_dataset.reshape(-1, 28, 28, 1),
 X_train.reshape(-1, 28, 28, 1),
 epochs=10,
 batch_size=200,
 validation_data=(noise_test_set.reshape(-1, 28, 28,
1), X_test.reshape(-1, 28, 28, 1)))
STEP #5: EVALUATE TRAINED MODEL
PERFORMANCE
Code evaluation = autoencoder.evaluate(noise_test_set.reshape(-1, 28, 28,
1), X_test.reshape(-1, 28, 28, 1))
print('Test Accuracy : {:.3f}'.format(evaluation))
o/p 313/313 [==============================] - 1s 4ms/step - loss: 0.302
2
Test Accuracy : 0.302
Code predicted = autoencoder.predict(noise_test_set[:10].reshape(-1, 28,
28, 1))
Code predicted.shape
o/p (10, 28, 28, 1)
Code fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True,
sharey=True, figsize=(20,4))
for images, row in zip([noise_test_set[:10], predicted], axes):
 for img, ax in zip(images, row):
 ax.imshow(img.reshape((28, 28)), cmap='Greys_r')
 ax.get_xaxis().set_visible(False)
 ax.get_yaxis().set_visible(False)
 --------------------
5

# Imports required packages
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
Code Loading and Preparing Data
# Loads fashion mnist dataset
fashion = tf.keras.datasets.fashion_mnist.load_data()
Department of CSE, DSCE Deep Learning Lab - VI semester
# Each training and test example is assigned to one of
the following labels.
class_names = ["T-shirt/top", "Trouser", "Pullover",
"Dress", "Coat", "Sandal", \
 "Shirt", "Sneaker", "Bag", "Ankle boot"]
# Considering dataset is organized in tuple, items are
referenced as follows
(X_train_full, y_train_full), (X_test, y_test) = fashion
# Checks the shape of the datasets
print("Train dataset shape:", X_train_full.shape)
print("Test dataset shape:", X_test.shape)
o/p Train dataset shape: (60000, 28, 28)
Test dataset shape: (10000, 28, 28)
Code # Checks the data type of the data
X_train_full.dtype
o/p dtype('uint8')
Code # Considering the data type of the data, it normalizes the data
between 0 and 1
# to make neural network model training efficient
X_train_full, X_test = X_train_full / 255., X_test / 255.
# Prints the labels for refer to the class index
y_train_full
o/p array([9, 0, 0, ..., 3, 0, 5], dtype=uint8)
Note: Considering the target binary classification model is expected
to classify "Pullover" and "T-shirt/top", it separates data for these
two classes leaving data for remaining 8 classes to build a model to
be considered as pretrained model later.
Code # Finds the index for the target class "Pullover" and "T-shirt/top"
as
# dataset labels contains class indexes instead of class names
class_0_index = class_names.index("Pullover")
class_1_index = class_names.index("T-shirt/top")
print("Index of class_0:", class_0_index)
print("Index of class_1:", class_1_index)
o/p Index of class_0: 2
Index of class_1: 0
Code # Gets the indexes of training label containing either classes
class_0_1_index_flag = [True if (x==class_0_index or
x==class_1_index) else False for x in y_train_full]
# Shows few flags
print(class_0_1_index_flag[:10])
o/p [False, True, True, False, True, True, False, True, False, False]
Code # Seperates dataset containing data for two classes
X_train_2_classes_full = X_train_full[class_0_1_index_flag]
# Checks the shape of the dataset
X_train_2_classes_full.shape
o/p (12000, 28, 28)
Code # Flips bool values (True to False and False to True) to get the
flags against
# other classes in the training label
class_0_1_index_flag_flipped = [not flag for flag in
class_0_1_index_flag]
# Shows few flags
print(class_0_1_index_flag_flipped[:10])
o/p [True, False, False, True, False, False, True, False, True, True]
Code # Seperates dataset containing data for the remaining 8 classes
X_train_8_classes_full = X_train_full[class_0_1_index_flag_flipped]
# Checks the shape of the dataset
X_train_8_classes_full.shape
o/p (48000, 28, 28)
Code # Sum of the first dimension value of both the dataset should be
equal to the total number of training instances
X_train_2_classes_full.shape[0] + X_train_8_classes_full.shape[0]
o/p 60000
Code # Similarly, separates targets to contain only respective labels
y_train_2_classes_full = y_train_full[class_0_1_index_flag]
y_train_8_classes_full = y_train_full[class_0_1_index_flag_flipped]
# Checks the shape of the targets
print(y_train_2_classes_full.shape)
print(y_train_8_classes_full.shape)
o/p (12000,)
(48000,)
 # Separates validation dataset
X_train_8_classes, X_val_8_classes, y_train_8_classes,
y_val_8_classes = train_test_split(
 X_train_8_classes_full, y_train_8_classes_full, test_size=5000,
random_state=42, stratify=y_train_8_classes_full)
# Prints the shape of the separated datasets both containing 8
classes
print(X_train_8_classes.shape)
print(X_val_8_classes.shape)
o/p (43000, 28, 28)
(5000, 28, 28)
Code # Then standardizes the datasets by first calculating mean and
standard deviation, and then
# by subtracting the mean from the data and then dividing the data by
standard deviation
pixel_means_8_classes = X_train_8_classes.mean(axis=0, keepdims=True)
pixel_stds_8_classes = X_train_8_classes.std(axis=0, keepdims=True)
 of CSE, DSCE Deep Learning Lab - VI semester
X_train_8_classes_scaled = (X_train_8_classes -
pixel_means_8_classes) / pixel_stds_8_classes
X_val_8_classes_scaled = (X_val_8_classes - pixel_means_8_classes) /
pixel_stds_8_classes
# As the labels ranges from [1, 3, 4, 5, 6, 7, 8, 9], it normalizes
the label from 0 through 7
label_encoder_8_classes = LabelEncoder()
y_train_8_classes_encoded =
label_encoder_8_classes.fit_transform(y_train_8_classes)
y_val_8_classes_encoded =
label_encoder_8_classes.transform(y_val_8_classes)
# Initializes the following densed neural network with arbirary
number of layers and compiles it
model = tf.keras.Sequential([
 tf.keras.layers.Flatten(input_shape=[28, 28]),
 tf.keras.layers.Dense(100, activation="relu",
kernel_initializer="he_normal"),
 tf.keras.layers.Dense(100, activation="relu",
kernel_initializer="he_normal"),
 tf.keras.layers.Dense(100, activation="relu",
kernel_initializer="he_normal"),
 tf.keras.layers.Dense(8, activation="softmax")
])
model.compile(
 loss="sparse_categorical_crossentropy",
 optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
 metrics=["accuracy"])
# Checks for model summary [optional]
model.summary()
# Fits the model over specific number iterations (epochs) and
validation data
# to observe the learning performance during training
model_history = model.fit(X_train_8_classes_scaled,
y_train_8_classes_encoded, epochs=20,
 validation_data=(X_val_8_classes_scaled,
y_val_8_classes_encoded))
o/p Epochs running
…
…
…
Code # Saves the trained model on disk to be used as pretrained model
later.
# NOTE: Folder "model" must exist for model file to be saved into.
model.save("./models/my_fashion_mnist_model.keras")
Note: Training Target Model from Scratch
Preprocesses Datasets
Code # Separates validation dataset from the data containg 2 classes
X_train_2_classes, X_val_2_classes, y_train_2_classes,
y_val_2_classes = train_test_split(
 X_train_2_classes_full, y_train_2_classes_full, test_size=3000,
random_state=42, stratify=y_train_2_classes_full)
Code # Prints the shape of the separated datasets containing both classes
print(X_train_2_classes.shape)
print(X_val_2_classes.shape)
o/p (9000, 28, 28)
(3000, 28, 28)
e # Then standardizes the datasets by first calculating mean and
standard deviation, and then
# by subtracting the mean from the data and then dividing the data by
standard deviation
pixel_means_2_classes = X_train_2_classes.mean(axis=0, keepdims=True)
pixel_stds_2_classes = X_train_2_classes.std(axis=0, keepdims=True)
X_train_2_classes_scaled = (X_train_2_classes -
pixel_means_2_classes) / pixel_stds_2_classes
X_val_2_classes_scaled = (X_val_2_classes - pixel_means_2_classes) /
pixel_stds_2_classes
Code # As the labels ranges from [1, 3, 4, 5, 6, 7, 8, 9], it normalizes
the label from 0 through 7
label_encoder_2_classes = LabelEncoder()
y_train_2_classes_encoded =
label_encoder_2_classes.fit_transform(y_train_2_classes)
y_val_2_classes_encoded =
label_encoder_2_classes.transform(y_val_2_classes)
Code # Clears the name counters and
# sets the global random seed for operations that rely on a random
seed
tf.keras.backend.clear_session()
tf.random.set_seed(42)
# Initializes the following densed neural network with arbirary
number of layers and compiles it
model_from_scratch = tf.keras.Sequential([
 tf.keras.layers.Flatten(input_shape=[28, 28]),
 tf.keras.layers.Dense(100, activation="relu",
kernel_initializer="he_normal"),
 tf.keras.layers.Dense(100, activation="relu",
kernel_initializer="he_normal"),
 tf.keras.layers.Dense(100, activation="relu",
kernel_initializer="he_normal"),
 tf.keras.layers.Dense(1, activation="sigmoid")
])
model_from_scratch.compile(
 loss="binary_crossentropy",
 optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
 metrics=["accuracy"])
Code # Checks for model summary [optional]
model_from_scratch.summary()
de # Fits the model over specific number iterations (epochs) on all the
training data available for the 2 classes
# and validation data to observe the learning performance during
training
model_from_scratch_history =
model_from_scratch.fit(X_train_2_classes_scaled,
y_train_2_classes_encoded, epochs=20,

validation_data=(X_val_2_classes_scaled, y_val_2_classes_encoded))
o/p Epochs running
…
…
…
# Gets the indexes of test label containing either classes
class_0_1_index_flag = [True if (x==class_0_index or
x==class_1_index) else False for x in y_test]
Department of CSE, DSCE Deep Learning Lab - VI semester
Code # Seperates dataset containing data for two classes from the whole
test set also containing other classes
X_test_2_classes = X_test[class_0_1_index_flag]
# Checks the shape of the dataset
X_test_2_classes.shape
 (2000, 28, 28)
Code # Similarly, separates targets to contain only respective labels
y_test_2_classes = y_test[class_0_1_index_flag]
# Normalizes the test labels for the 2 classes using the already
fitted encoder
y_test_2_classes_encoded =
label_encoder_2_classes.transform(y_test_2_classes)
# Prints the encoded classes for reference
y_test_2_classes_encoded
o/p array([1, 1, 0, ..., 0, 0, 1])
Code # Standardizes the test set by subtracting the mean from the data and
then dividing the data by standard deviation
X_test_2_classes_scaled = (X_test_2_classes - pixel_means_2_classes)
/ pixel_stds_2_classes
# Evaluates the test prediction performance on the model built from
scratch
model_from_scratch.evaluate(X_test_2_classes_scaled,
y_test_2_classes_encoded)
o/p 63/63 ━━━━━━━━━━━━━━━━━━━━ 0s 921us/step - accuracy: 0.9
640 - loss: 0.1104
[0.1123715341091156, 0.9620000123977661]
NOTE: The above model that was built from scratch over 9000 [12000
total - 3000 validation instances] training instances containing d
ata for 2 classes, reached 96.20% accuracy in test set. The experi
ment continues to apply transfer learning by reusing pretrained la
yers from first model built over other 8 classes to check if new m
odel trained over less data can achieve accuracy from the model bu
ilt from scratch.
e Transfer Learning
# Loads the saved model created to be used as pretrained model
model_using_pretrained_layers =
tf.keras.models.load_model("./models/my_fashion_mnist_model.keras")
# Checks the model summary especially to refer to the last layer i.e.
the output layer
model_using_pretrained_layers.summary()
# Removes the last layer (containing 8 output) to add task specific
binary output layer
model_using_pretrained_layers.pop()
# And then adds a binary output layer
model_using_pretrained_layers.add(tf.keras.layers.Dense(1,
activation="sigmoid", name="output"))
# Then verifies the same visualizing the model summary
model_using_pretrained_layers.summary()
 Fine-tuning already pretrained model
# Considers only 60% of the 2-classes training set to check the
effectiveness of the transfer learning
X_train_2_classes_scaled_subset, _, y_train_2_classes_encoded_subset,
_ = train_test_split(
Department of CSE, DSCE Deep Learning Lab - VI semester
 X_train_2_classes_scaled, y_train_2_classes_encoded,
train_size=0.60, stratify=y_train_2_classes_encoded)
# First sets all the pretrained layers (except for the newly added
output layer) non-trainable
for layer in model_using_pretrained_layers.layers[:-1]:
 layer.trainable = False
# Then trains the just the output layer
tf.keras.backend.clear_session()
tf.random.set_seed(42)
model_using_pretrained_layers.compile(
 loss="binary_crossentropy",
optimizer=tf.keras.optimizers.SGD(learning_rate=0.001))
model_using_pretrained_layers_history =
model_using_pretrained_layers.fit(
 X_train_2_classes_scaled_subset,
y_train_2_classes_encoded_subset, epochs=5,
 validation_data=(X_val_2_classes_scaled,
y_val_2_classes_encoded))
o/p Epochs running
…
…
…
Code # Now, makes all the pretrained layers trainable and performs
retraining over small smaller
# learning rate for longer iterations
for layer in model_using_pretrained_layers.layers[:-1]:
 layer.trainable = True
# Recompiles the model due to change of trainability of the layers
model_using_pretrained_layers.compile(
 loss="binary_crossentropy",
optimizer=tf.keras.optimizers.SGD(learning_rate=0.001))
model_using_pretrained_layers_history =
model_using_pretrained_layers.fit(
 X_train_2_classes_scaled_subset, y_train_2_classes_encoded_subset,
epochs=100,
 validation_data=(X_val_2_classes_scaled, y_val_2_classes_encoded))
o/p Epochs running
…
…
…
Code # Evaluates the test prediction performance on the model built using
pretrained layers
Department of CSE, DSCE Deep Learning Lab - VI semester
model_using_pretrained_layers.evaluate(X_test_2_classes_scaled,
y_test_2_classes_encoded)
o/p 63/63 ━━━━━━━━━━━━━━━━━━━━ 0s 798us/step - accuracy: 0.9
687 - loss: 0.0929
[0.09692149609327316, 0.9674999713897705]
NOTE: Though this model built over pretrained layers using on 60%
of the available training set, but could also achieved 96.75% test
accuracy as compared to 96.2% accuracy of the model built from scr
atch over the full training set. The error rate was improved by 14
% [(96.75−96.2)÷(100−96.20)×100].
