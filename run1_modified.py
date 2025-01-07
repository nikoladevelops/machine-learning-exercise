import tensorflow as tf
data = tf.keras.datasets.fashion_mnist

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

# Този път увеличаваме епохите да са 10, вместо 5
model.fit(training_images, training_labels, epochs=10)

# Оценка на модела на тестови данни
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"Точност на теста: {test_accuracy*100:.2f}%")

# Прогноза
classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])