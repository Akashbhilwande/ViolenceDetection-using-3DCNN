import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPool3D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# Parameters
SEQUENCE_LENGTH = 20  # Number of frames per video clip
IMG_SIZE = 112        # Resize frames to 112x112
BATCH_SIZE = 8
EPOCHS = 30
NUM_CLASSES = 2

def load_video_frames(video_path, sequence_length=SEQUENCE_LENGTH, img_size=IMG_SIZE):
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = max(total_frames // sequence_length, 1)
    
    for i in range(sequence_length):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * frame_step)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (img_size, img_size))
            frame = frame / 255.0  # Normalize
            frames.append(frame)
    cap.release()
    
    # Pad with black frames if video is too short
    while len(frames) < sequence_length:
        frames.append(np.zeros((img_size, img_size, 3)))
    
    return np.array(frames)

def load_dataset(dataset_path):
    X, y = [], []
    classes = ['Nonviolence', 'Violence']
    
    for class_idx, class_name in enumerate(classes):
        class_path = os.path.join(dataset_path, class_name)
        for video_file in os.listdir(class_path):
            video_path = os.path.join(class_path, video_file)
            frames = load_video_frames(video_path)
            X.append(frames)
            y.append(class_idx)
    
    return np.array(X), to_categorical(np.array(y), num_classes=NUM_CLASSES)

# Load dataset
dataset_path = r'C:\Users\bhilw\OneDrive\Documents\violence-detection-main\violence-detection-main\dataset'  # Update this path
X, y = load_dataset(dataset_path)

# Split dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Model architecture
model = Sequential([
    Conv3D(32, (3, 3, 3), activation='relu', input_shape=(SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE, 3)),
    MaxPool3D(pool_size=(1, 2, 2)),
    Conv3D(64, (3, 3, 3), activation='relu'),
    MaxPool3D(pool_size=(1, 2, 2)),
    Conv3D(128, (3, 3, 3), activation='relu'),
    MaxPool3D(pool_size=(1, 2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

# Compile model
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy', 
                       tf.keras.metrics.Precision(name='precision'),
                       tf.keras.metrics.Recall(name='recall')])

# Callbacks
checkpoint = ModelCheckpoint('best_model.h5', 
                             monitor='val_accuracy',
                             save_best_only=True,
                             mode='max')

early_stop = EarlyStopping(monitor='val_loss',
                           patience=10,
                           restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=3,
                              min_lr=1e-6)

# Train model
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    callbacks=[checkpoint, early_stop, reduce_lr])

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

model.load_weights('best_model.h5')  # Load best model
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