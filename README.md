# MNIST Digit Classification using Convolutional Neural Network (CNN)
This project builds a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset. The model is trained on the MNIST dataset and saved for future use. Additionally, a function is provided to load the model and predict a digit from a new grayscale image.

## Project Structure
- mnist_model.keras: The trained CNN model that is saved after training.
- Script: Python script for training, evaluating, and saving the model. It also includes a function to predict digits from new images.

## Prerequisites
### Python Libraries
You will need the following Python libraries installed to run this project:
- tensorflow
- numpy
- matplotlib
- PIL (comes with tensorflow.keras.preprocessing.image)
You can install the required libraries using the following commands:
```
bash

pip install tensorflow numpy matplotlib
```
### Dataset
The project uses the MNIST dataset, which contains 60,000 training images and 10,000 test images of handwritten digits from 0 to 9. The dataset is automatically downloaded when running the code.

## Code Overview
### Step 1: Load and Preprocess the MNIST Dataset
The script starts by loading the MNIST dataset using tensorflow.keras.datasets. It normalizes the pixel values of the images to the range [0, 1] and reshapes the images to (28, 28, 1) to match the input requirements of the CNN.

### Step 2: Build the Convolutional Neural Network Model
A CNN model is built using tensorflow.keras.models.Sequential. The model consists of the following layers:
- 2D Convolution Layer (32 filters, 3x3 kernel)
- MaxPooling Layer (2x2 pool size)
- 2D Convolution Layer (64 filters, 3x3 kernel)
- MaxPooling Layer (2x2 pool size)
- Flatten Layer (to convert the 2D matrices into a 1D vector)
- Dense Layer (64 units, ReLU activation)
- Dense Layer (10 units, Softmax activation for multi-class classification)

### Step 3: Compile and Train the Model
The model is compiled with the adam optimizer and sparse_categorical_crossentropy loss function, which is suitable for multi-class classification. The model is trained for 10 epochs using the training data (x_train and y_train), and 10% of the training data is used for validation.

### Step 4: Evaluate the Model
After training, the model is evaluated on the test data (x_test and y_test) to calculate its accuracy.

### Step 5: Save the Model
The trained model is saved to a file (mnist_model.keras) using the model.save() function. This allows the model to be loaded later for inference without the need to retrain it.

### Step 6: Load and Predict a New Image
A function load_and_predict_image(image_path) is provided to load a saved model and predict the digit from a new grayscale image. The image is resized to 28x28 and normalized before prediction. The predicted digit is returned as output.

### Example Usage
To predict a digit from a new image, you can replace the image path with the path to your image file in the following line:
```
python

predicted_digit = load_and_predict_image('C:/Users/User/Desktop/images for testing/image 2.png')
print(f'The predicted digit is: {predicted_digit}')
```
Make sure the input image is:
- A grayscale image.
- Sized 28x28 pixels (the script resizes it automatically).

## How to Run the Script
1. Clone or download this repository.
2. Install the required libraries.
3. Run the script using a Python environment:
```
bash

python mnist_cnn.py
```
4. Once the model is trained and saved, you can predict digits from new images using the load_and_predict_image() function.

## Example Workflow:
- Train and save the model: This will automatically happen when running the script.
- Evaluate the model: The model's test accuracy will be printed after training.
- Predict new images: Use the provided function load_and_predict_image() to predict digits from new images.

## License

[MIT](https://choosealicense.com/licenses/mit/)

### Screenshot
![Logo](https://github.com/Junyao95/MNIST-Digit-Classification-using-Convolutional-Neural-Network-CNN-/blob/main/DhKHDNzS0OJhH2uiMOkl_mnist.png?raw=true)
