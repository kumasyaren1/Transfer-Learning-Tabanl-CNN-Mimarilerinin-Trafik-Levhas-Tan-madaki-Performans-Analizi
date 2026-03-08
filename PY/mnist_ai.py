import tensorflow as tf
from tensorflow import keras

# veri seti
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# veriyi normalize et
x_train = x_train / 255.0
x_test = x_test / 255.0

# model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

# model ayarları
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# eğit
model.fit(x_train, y_train, epochs=5)

# test
test_loss, test_acc = model.evaluate(x_test, y_test)

print("Test doğruluğu:", test_acc)