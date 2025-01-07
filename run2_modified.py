import tensorflow as tf
data = tf.keras.datasets.fashion_mnist

# Създаване на Callback за ранно спиране при достигане на точност над 90% (този път искаме по голяма точност сравнение с оригиналното)
class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.90:
            print("\n Достигнахме 90% точност, спираме обучението!")
            self.model.stop_training = True

callbacks = MyCallback()

(training_images, training_labels), (test_images, test_labels) = data.load_data()

# Рескалиране на изображенията и подготвяне за моделиране
training_images = training_images.reshape(60000, 28, 28, 1) / 255.0
test_images = test_images.reshape(10000, 28, 28, 1) / 255.0

# Създаване на по-сложен модел с допълнителни слоеве
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Компилация на модела
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Обучение на модела с Callback за ранно спиране и увеличение на броя епохи
model.fit(training_images, training_labels, epochs=10, callbacks=[callbacks])

# Оценка на модела на тестови данни
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"Точност на теста: {test_accuracy*100:.2f}%")

# Прогноза
classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])