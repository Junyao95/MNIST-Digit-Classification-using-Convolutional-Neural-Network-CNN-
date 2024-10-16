import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from tensorflow.keras import layers, models 
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing import image 

# step 1: load and preprocess the MNIST Dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the images to [0, 1] range
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape the images to (28, 28, 1) for the convolutional layers
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Step 2: Build the Neural Network Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)), 
    layers.Flatten(), 
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax') # 10 classes for digits 0-9
])

# Complie the model 
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
)

# Step 3: Train the Model
model.fit(x_train, y_train, epochs=10, validation_split=0.1)

# Step 4: Evaluate the model on test data
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc:.4f}')

# Step 5: Save the Model
model.save('mnist_model.keras')
print("Model saved as 'mnist_model.keras'")

# Step 6: Load a new Image and Predict
def load_and_predict_image(image_path):
    model = load_model('mnist_model.keras')
    img = image.load_img(image_path, target_size=(28, 28), color_mode='grayscale')
    img_array = image.img_to_array(img) / 255.0 # normalize the image
    img_array = np.expand_dims(img_array, axis=0) # add bath dimension
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction) # get the class with the highest probability
    return predicted_class

# Example Usage
# replace 'your_image'.png' with the path to your image file
predicted_digit = load_and_predict_image('C:/Users/User/Desktop/images for testing/image 2.png')
print(f'The predicted digit is: {predicted_digit}')
