# CLASIFICACIÓN DE DÍGITOS MANUSCRITOS – CNN (adaptado del código Perros vs Gatos)
# Autor: [Tu nombre]
# Curso: Mantenimiento de Equipos Periféricos

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import itertools

# --- 1) CARGA DE DATOS ---
(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train[:80%]', 'train[80%:]'],
    as_supervised=True,
    with_info=True
)

# --- 2) PREPROCESAMIENTO ---
def preprocessing(img, label):
    img = tf.image.resize(img, (28, 28))
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

train_ds = ds_train.map(preprocessing).shuffle(10000).batch(32).prefetch(1)
test_ds = ds_test.map(preprocessing).batch(32).prefetch(1)

# --- 3) VISUALIZACIÓN DE MUESTRAS ---
for imgs, labels in train_ds.take(1):
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.imshow(tf.squeeze(imgs[i]), cmap='gray')
        plt.title(f'Dígito: {labels[i].numpy()}')
        plt.axis('off')
plt.show()

# --- 4) MODELO CNN ---
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.summary()

# --- 5) COMPILACIÓN Y ENTRENAMIENTO ---
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_ds, epochs=5, validation_data=test_ds)

# --- 6) GRAFICAR PRECISIÓN ---
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión del Modelo')
plt.grid(); plt.legend(); plt.show()

# --- 7) EVALUACIÓN ---
test_loss, test_acc = model.evaluate(test_ds)
print(f"Precisión en test: {test_acc:.4f}")

# --- 8) MATRIZ DE CONFUSIÓN ---
y_true, y_pred = [], []
for img_batch, label_batch in test_ds:
    preds = model.predict(img_batch)
    y_true.extend(label_batch.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

cm = confusion_matrix(y_true, y_pred)
print(classification_report(y_true, y_pred, digits=4))

plt.imshow(cm, cmap='Blues')
plt.title("Matriz de Confusión")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.colorbar()
plt.show()

# --- 9) PREDICCIÓN VISUAL ---
for img, label in test_ds.take(1):
    preds = model.predict(img)
    i = 10
    plt.imshow(tf.squeeze(img[i]), cmap='gray')
    plt.title(f"Real: {label[i].numpy()} | Predicho: {np.argmax(preds[i])}")
    plt.axis('off')
    plt.show()
