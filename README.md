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

auto
----
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
# Load MNIST data
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
# Add noise
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)
# Build autoencoder
input_img = Input(shape=(28, 28, 1))

# Encoder
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# Decoder
x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
# Train
autoencoder.fit(x_train_noisy, x_train,
                epochs=5,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test_noisy, x_test))
# Visualize
decoded_imgs = autoencoder.predict(x_test_noisy)
n = 15
plt.figure(figsize=(20, 4))
for i in range(n):
    # Noisy input
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test_noisy[i].reshape(28, 28), cmap='gray')
    ax.axis('off')

    # Cleaned output
    ax = plt.subplot(3, n, i + n + 1) 
    plt.imshow(decoded_imgs[i].reshape(28, 28), cmap='gray')
    ax.axis('off')

    # Ground truth
    ax = plt.subplot(3, n, i + 2*n + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    ax.axis('off')

plt.tight_layout()
plt.show()


----------------

7

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset from CSV
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv'
df = pd.read_csv(url, parse_dates=['Date'])

# Plot to visualize
plt.figure(figsize=(10, 4))
plt.plot(df['Temp'])
plt.title("Daily Min Temperatures")
plt.xlabel("Days")
plt.ylabel("Temperature (°C)")
plt.show()
# Convert to NumPy array
temps = df['Temp'].values.astype(np.float32)

# Normalize for better training
mean = temps.mean()
std = temps.std()
temps = (temps - mean) / std

# Function to create sequences
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

# Create input/output sequences
window_size = 7  # Predict next day from previous 7 days
X, y = create_sequences(temps, window_size)

# Reshape for RNN input: (samples, time steps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split into train/test sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

model = Sequential([
    SimpleRNN(64, return_sequences=True, activation='tanh', input_shape=(window_size, 1)),
    SimpleRNN(32, activation='tanh'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.summary()
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop]
)
# Predict on test data
y_pred = model.predict(X_test)

# Denormalize predictions and actual values
y_pred_denorm = y_pred * std + mean
y_test_denorm = y_test * std + mean

# Plot actual vs predicted
plt.figure(figsize=(10, 4))
plt.plot(y_test_denorm, label='Actual')
plt.plot(y_pred_denorm, label='Predicted')
plt.legend()
plt.title("Actual vs Predicted Temperatures")
plt.show()
-----------------------

8

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv'
df = pd.read_csv(url, parse_dates=['Date'])
# Normalize temperatures
temps = df['Temp'].values.astype(np.float32)
mean = temps.mean()
std = temps.std()
temps = (temps - mean) / std
# Create sequences
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

window_size = 7
X, y = create_sequences(temps, window_size)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Train-test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential([
    LSTM(64, activation='tanh', input_shape=(window_size, 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.summary()
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop]
)
# Predict
y_pred = model.predict(X_test)

# Denormalize
y_pred_denorm = y_pred * std + mean
y_test_denorm = y_test * std + mean

# Plot
plt.figure(figsize=(10, 4))
plt.plot(y_test_denorm, label='Actual')
plt.plot(y_pred_denorm, label='Predicted')
plt.title('LSTM: Actual vs Predicted Temperature')
plt.xlabel('Days')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()


----------------------


from google.colab import drive
drive.mount('/content/drive')
import os
import numpy as np
import pandas as pd
import random
from zipfile import ZipFile
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.applications import VGG16, ResNet50V2
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.metrics import Precision,Recall
from sklearn.metrics import accuracy_score,precision_score,classification_report,f1_score,confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
with ZipFile('/content/drive/MyDrive/Copy of MangoLeaf (1).zip','r')as ZipObj:
  ZipObj.extractall('/content/drive/MyDrive')
  import zipfile
import os

zip_path = '/content/drive/MyDrive/Copy of MangoLeaf (1).zip'
extract_path = '/content/drive/MyDrive'

# Extract the zip file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
import splitfolders

os.chdir('/content/drive/MyDrive/MangoLeaf')

#input = where dataset is present
#output = where you want the split datasets saved.

splitfolders.ratio("/content/drive/MyDrive/MangoLeaf", output="/content/drive/MyDrive", seed=1337, ratio=(.75, .1, .15))

# ratio of split are in order of train/val/test.

os.chdir('../../')
print("Train classes:", os.listdir('/content/drive/MyDrive/train'))
train_datagen = ImageDataGenerator(rescale=1./255,horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_dir = '/content/drive/MyDrive/train'
train_generator = train_datagen.flow_from_directory(train_dir,
                                                   class_mode = 'categorical',
                                                   target_size = (224,224),
                                                   batch_size = 10)

validation_dir = '/content/drive/MyDrive/val'
validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                             class_mode = 'categorical',
                                                             target_size = (224,224),
                                                             batch_size = 10)

test_dir = '/content/drive/MyDrive/test'
test_generator = test_datagen.flow_from_directory(test_dir,
                                                             class_mode = 'categorical',
                                                             target_size = (224,224),
                                                             batch_size = 10)
                                                             pre_trained_model = ResNet50V2(input_shape=(224,224,3),include_top=False,weights='imagenet')

for layer in pre_trained_model.layers:
    layer.trainable = False
    pre_trained_model.summary()
    x = pre_trained_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(4, activation='softmax')(x)
model = Model(pre_trained_model.input, outputs=predictions)
model.compile(optimizer = Adam(learning_rate=0.0001),loss='categorical_crossentropy',metrics=['Accuracy'])
history = model.fit(train_generator,epochs=2,validation_data=validation_generator,batch_size=64)
loss, acc = model.evaluate(test_generator)
print(f'Test Accuracy: {acc:.4f}')
# Predict the class probabilities
y_pred_probs = model.predict(test_generator)

# Convert probabilities to class indices
y_pred = np.argmax(y_pred_probs, axis=1)

# Get true labels
y_true =test_generator.classes

# Get class labels
class_labels = list(test_generator.class_indices.keys())
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
print("Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_labels))
precision = precision_score(y_true, y_pred, average='macro')
print(f'Macro Average Precision: {precision:.4f}')
from tensorflow.keras.preprocessing import image

img_path = '/content/drive/MyDrive/test/4Healthy/Not_Infected_1215.jpg'
img = image.load_img(img_path, target_size=(224,224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)

prediction = model.predict(img_array)
predicted_class = class_labels[np.argmax(prediction)]
print("Predicted Class:", predicted_class)
