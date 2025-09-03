import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from sklearn.metrics import accuracy_score
import warnings
from tensorflow.keras.preprocessing.image import ImageDataGenerator

warnings.filterwarnings('ignore')

def load_dataset(data_path='data/Train'):
    print("Loading Traffic Sign Dataset...")
    data = []
    labels = []
    classes = 43

    for i in range(classes):
        class_path = os.path.join(data_path, str(i))
        if not os.path.exists(class_path):
            continue
        images = os.listdir(class_path)
        for image_file in images:
            try:
                image_path = os.path.join(class_path, image_file)
                image = Image.open(image_path).convert('RGB').resize((30, 30))
                data.append(np.array(image))
                labels.append(i)
            except Exception as e:
                print(f"Error loading {image_file}: {e}")
    
    data = np.array(data)
    labels = np.array(labels)
    return data, labels

def create_cnn_model(input_shape):
    model = Sequential([
        Conv2D(32, (5, 5), activation='relu', input_shape=input_shape),
        Conv2D(32, (5, 5), activation='relu'),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(43, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def plot_training_history(history):
    """Plot training accuracy and loss"""
    # --- NEW: Check for and create the 'outputs' directory ---
    output_dir = 'outputs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.legend()
    
    # Save the figure into the 'outputs' directory
    plt.savefig(os.path.join(output_dir, 'training_history_augmented.png'))
    plt.show()

def main():
    """Main training function"""
    print("=== Traffic Sign Recognition Training (with Data Augmentation) ===\n")
    if not os.path.exists('data/Train/0'):
        print("Dataset not found! Please check dataset structure.")
        return

    data, labels = load_dataset()
    data = data.astype('float32') / 255.0
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42, stratify=labels)
    y_train_cat = to_categorical(y_train, 43)
    y_test_cat = to_categorical(y_test, 43)

    model = create_cnn_model(X_train.shape[1:])
    model.summary()

    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        fill_mode='nearest'
    )
    datagen.fit(X_train)

    print("\nStarting training with augmented data...")
    history = model.fit(
        datagen.flow(X_train, y_train_cat, batch_size=64),
        epochs=30,
        validation_data=(X_test, y_test_cat),
        verbose=1
    )

    model_path = 'models/traffic_classifier_augmented.h5'
    model.save(model_path)
    print(f"\nAugmented model saved to {model_path}")

    plot_training_history(history)
    print("\n=== Training completed successfully! ===")

if __name__ == "__main__":
    main()
