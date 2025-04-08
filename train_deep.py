import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv3D, MaxPool3D, BatchNormalization,
    GlobalAveragePooling3D, Dense, Dropout,
    SpatialDropout3D, TimeDistributed
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# Parameters
SEQUENCE_LENGTH = 50
IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 50
NUM_CLASSES = 2

def load_video_frames(video_path, sequence_length=SEQUENCE_LENGTH, img_size=IMG_SIZE, batch_size=10):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            return np.zeros((sequence_length, img_size, img_size, 3))
        
        indices = np.linspace(0, total_frames - 1, num=sequence_length, dtype=int)
        
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_frames = []
            
            for idx in batch_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, (img_size, img_size))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    batch_frames.append(frame / 255.0)
                else:
                    batch_frames.append(np.zeros((img_size, img_size, 3)))
            
            frames.extend(batch_frames)
            
        while len(frames) < sequence_length:
            frames.append(frames[-1] if frames else np.zeros((img_size, img_size, 3)))
        
    except Exception as e:
        print(f"Error processing {video_path}: {str(e)}")
        return None
    finally:
        cap.release()
        
    return np.array(frames, dtype=np.float32)


def load_dataset(dataset_path):
    X, y = [], []
    classes = ['Nonviolence', 'Violence']
    
    for class_idx, class_name in enumerate(classes):
        class_path = os.path.join(dataset_path, class_name)
        for video_file in os.listdir(class_path):
            video_path = os.path.join(class_path, video_file)
            frames = load_video_frames(video_path)
            if frames is not None:
                X.append(frames)
                y.append(class_idx)
    
    return np.array(X, dtype=np.float32), to_categorical(np.array(y), num_classes=NUM_CLASSES)

# Load dataset
dataset_path = r'C:\Users\bhilw\OneDrive\Documents\violence-detection-main\violence-detection-main\Real Life Violence Dataset Train'
X, y = load_dataset(dataset_path)

# Split dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# TimeDistributed data augmentation
data_augmentation = tf.keras.Sequential([
    TimeDistributed(tf.keras.layers.RandomFlip("horizontal")),
    TimeDistributed(tf.keras.layers.RandomRotation(0.05)),
    TimeDistributed(tf.keras.layers.RandomContrast(0.1)),
    TimeDistributed(tf.keras.layers.RandomZoom(0.1)),
])

# Improved model architecture
model = Sequential([
    tf.keras.layers.Input(shape=(SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE, 3)),
    data_augmentation,
    
    Conv3D(64, (3, 3, 3), activation='relu', padding='same',
           kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    MaxPool3D((2, 2, 2)),
    SpatialDropout3D(0.25),
    
    Conv3D(128, (3, 3, 3), activation='relu', padding='same',
           kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    MaxPool3D((2, 2, 2)),
    SpatialDropout3D(0.25),
    
    Conv3D(256, (3, 3, 3), activation='relu', padding='same',
           kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    MaxPool3D((2, 2, 2)),
    SpatialDropout3D(0.25),
    
    Conv3D(512, (3, 3, 3), activation='relu', padding='same',
           kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    GlobalAveragePooling3D(),
    
    Dense(512, activation='relu', kernel_regularizer=l2(1e-4)),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

# Optimizer with weight decay
optimizer = tf.keras.optimizers.AdamW(
    learning_rate=1e-4,
    weight_decay=1e-4
)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy', 
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall')]
)

# Enhanced callbacks
callbacks = [
    ModelCheckpoint('best_model.keras', 
                   monitor='val_accuracy',
                   save_best_only=True,
                   mode='max'),
    EarlyStopping(monitor='val_loss',
                 patience=15,
                 restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss',
                     factor=0.5,
                     patience=5,
                     min_lr=1e-6)
]

# Train model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks
)

model.save('best_model_CCTV.keras') 

# Visualization
def plot_training_curves(history):
    metrics = ['loss', 'accuracy', 'precision', 'recall']
    plt.figure(figsize=(15, 10))
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        plt.plot(history.history[metric], label='Training '+metric)
        plt.plot(history.history['val_'+metric], label='Validation '+metric)
        plt.title(metric.upper())
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.legend()
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

plot_training_curves(history)

# Evaluation
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

model.load_weights('best_model_CCTV.keras')
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_val, axis=1)

print("Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=['Nonviolence', 'Violence']))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Nonviolence', 'Violence'],
            yticklabels=['Nonviolence', 'Violence'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()

