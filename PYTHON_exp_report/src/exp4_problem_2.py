import pandas as pd
import numpy as np
import tensorflow as tf
from keras import models
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten 
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from rich import print


physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("[bold green]GPU is enabled and memory growth is set.")
    except RuntimeError as e:
        print(f"[bold red]Error setting GPU memory growth: {e}")
else:
    print("[bold red]No GPU found. The code will run on CPU.")

X_train = np.array(pd.read_csv('data\experiment_4\hand-writing\csvTrainImages 60k x 784.csv'))
y_train = np.array(pd.read_csv('data\experiment_4\hand-writing\csvTrainLabel 60k x 1.csv'))
X_test = np.array(pd.read_csv('data\experiment_4\hand-writing\csvTestImages 10k x 784.csv'))
y_test = np.array(pd.read_csv('data\experiment_4\hand-writing\csvTestLabel 10k x 1.csv'))

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float')

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, 28*28)).reshape(-1, 28, 28, 1)
X_test = scaler.transform(X_test.reshape(-1, 28*28)).reshape(-1, 28, 28, 1)

model = models.Sequential()
model.add(Conv2D(filters=8, kernel_size=(5, 5), padding='same', input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=16, kernel_size=(5, 5), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='sigmoid'))

model2 = models.Sequential()
model2.add(Conv2D(filters=8, kernel_size=(5, 5), padding='same', input_shape=(28, 28, 1), activation='relu'))
model2.add(Conv2D(filters=8, kernel_size=(5, 5), padding='same', activation='relu'))
model2.add(Conv2D(filters=16, kernel_size=(5, 5), padding='same', activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu'))
model2.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu'))
model2.add(Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu'))
model2.add(Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu'))
model2.add(Conv2D(filters=128, kernel_size=(5, 5), padding='same', activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Conv2D(filters=128, kernel_size=(5, 5), padding='same', activation='relu'))
model2.add(Conv2D(filters=512, kernel_size=(5, 5), padding='same', activation='relu'))
model2.add(Conv2D(filters=512, kernel_size=(5, 5), padding='same', activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))

model2.add(Flatten())
model2.add(Dense(128, activation='relu'))
model2.add(Dropout(0.25))
model2.add(Dense(10, activation='sigmoid'))

model3 = models.Sequential()
model3.add(Conv2D(filters=8, kernel_size=(7, 7), padding='same', input_shape=(28, 28, 1), activation='relu'))
model3.add(MaxPooling2D(pool_size=(2, 2)))
model3.add(Conv2D(filters=16, kernel_size=(7, 7), padding='same', activation='relu'))
model3.add(MaxPooling2D(pool_size=(2, 2)))
model3.add(Conv2D(filters=32, kernel_size=(7, 7), padding='same', activation='relu'))
model3.add(MaxPooling2D(pool_size=(2, 2)))
model3.add(Conv2D(filters=64, kernel_size=(7, 7), padding='same', activation='relu'))
model3.add(MaxPooling2D(pool_size=(2, 2)))

model3.add(Flatten())
model3.add(Dense(128, activation='relu'))
model3.add(Dropout(0.25))
model3.add(Dense(10, activation='sigmoid'))

for model_ in [model2]:
    model_.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    epochs = 50
    history = model_.fit(X_train, y_train, validation_split=0.2, epochs=epochs, batch_size=256, verbose=1)

    print(f"train accuracy: {history.history['accuracy'][epochs - 1]:.7f}")

    test_loss, test_acc = model_.evaluate(X_test, y_test, verbose=2)
    print(f"test accuracy: {test_acc:.4f}")

    i = 1
    plt.grid(True)
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.xlabel('Epoch', style='italic')
    plt.ylabel('Loss', style="italic")
    plt.legend()
    plt.savefig(f'img/problem4-2-loss-model{i}.png', dpi=300)

    plt.grid(True)
    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation accuracy')
    plt.xlabel('Epoch', style='italic')
    plt.ylabel('Accuracy', style='italic')
    plt.legend()
    plt.savefig(f'img/problem4-2-acc-model{i}.png', dpi=300)
    i += 1
