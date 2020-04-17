# [Course 1 - Intro to Tensorflow](https://www.coursera.org/learn/introduction-tensorflow/home/welcome)

## [Week 1](https://www.coursera.org/learn/introduction-tensorflow/home/week/1)

### Primer
```mermaid
graph LR
Rules --> A[Traditional Programming]
Data  --> A
A --> Answers
```
#### Traditional Programming
- rules based programming
- if-then statements

```mermaid
graph LR
Answers --> B[Machine Learning]
Data  --> B
B --> Rules
```
#### Machine Learning
- lots of examples + label
- rules are inferred by ML

### Hello World - Neural Network
[colab](https://github.com/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%202%20-%20Lesson%202%20-%20Notebook.ipynb)

> X = 
> Y = 
Simple linear curve fitting  

> keras.Sequential()
Define successive layers

> keras.layers.Dense()
Define a layer of connected neurons 
this could be done as a list of Dense() inside Sequential() or Sequential.add(Dense())

> model.compile(optimizer='sgd', loss='mean_squared_error')
define the loss function, and optimizer - to guess the next value (for gradient descent)

> model.fit(epochs=N)
actual training of the model

> model.predict([input])
inference / find predicted values

> model.evaluate(test_images, test_labels)
evaluate the test set


## [Week 2](https://www.coursera.org/learn/introduction-tensorflow/home/week/2)

[colab](https://github.com/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%204%20-%20Lesson%202%20-%20Notebook.ipynb)

### Intro to Computer Vision
- Help computers look at the images (and understand the content)
- [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) -  78k images, 10 categories, 28x28 pixels each
- keras has in-build [datasets] (https://www.tensorflow.org/api_docs/python/tf/keras/datasets) including fashion MNIST

> fashion_mnist = keras.datasets.fashion_mnist
> (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

- separated training and test data
- labels are enum (numbers) instead of strings - to reduce language [bias](https://developers.google.com/machine-learning/fairness-overview/)

### Coding a CV Neural Network
> keras.layers.Flatten(input_shape=(28, 28))
Takes the input as a linear array

> keras.layers.Dense(128, activation=tf.nn.relu)
Hidden Layer

 > keras.layers.Dense(128, activation=tf.nn.softmax)
> Output layer

### Callback to terminate the training
```
class myCallback(keras.callbacks.Callback):
def on_epoch_end(self, epoch, logs={}):
    if logs.get('loss') < 0.4:
        self.model.stop_training = True

callbacks = myCallback()
model.fit(..., callbacks=[callbacks])
```
### Colab Exporations
- Simple NN (28x28 Flatten, 128 relu, 10 softmax) - no CNN

[Ex 1](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%204%20-%20Lesson%202%20-%20Notebook.ipynb#scrollTo=rquQqIx4AaGR) - What are the numbers?
A - It is the probability that each item is one of the 10 class (and using softmax, we choose the highest probability)

[Ex 2](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%204%20-%20Lesson%202%20-%20Notebook.ipynb#scrollTo=OgQSIfDSOWv6) Impact of larger neurons in the hidden layer(512)?
A- More neurons = slower (and more accurate) training; till an extend.

[Ex 5](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%204%20-%20Lesson%202%20-%20Notebook.ipynb#scrollTo=-0lF5MuvSuZF) What is the affect of adding additional hidden layer?
A - Usually more helpful but in this particular case, adding an additional layer reduced the accuracy.

## [Week 3](https://www.coursera.org/learn/introduction-tensorflow/home/week/3)

[colab](https://colab.sandbox.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%206%20-%20Lesson%202%20-%20Notebook.ipynb)

### CNN - Convolutions and Pooling
- Convolutions are like filters to extract features from images
- Pooling is compressing the result of convolution while reducing the image dimension

> tf.keras.layers.Conv2D(num_conv, (3, 3), activation='relu', input_shape=(28, 28, 1))
> tk.keras.layers.MaxPooling2D(2, 2)
Stack the Conv2D and MaxPooling2D before the regular NN layers.

> model.summary()
Shows the NN
the size of image is smaller in the layers because of conv-layers window

[Examples of filters](https://lodev.org/cgtutor/filtering.html)
<!--stackedit_data:
eyJoaXN0b3J5IjpbODIyODkyNTMwLC01NTcyNTkwNzMsNzk2OT
YxNTIyLC0xMTAxOTYyMTE2LC0xMDI1MzIyNTgxLC0xNzY0NDQ5
MzAyLDYxMTg2NDI0OCw3MjAxMzM1NTAsMzU4MTgzNzY4LC0xOD
U0NjUzNDcxLDE3MzUwMTY4MDMsNjkyMzI0MDQ0LDE1NzM2MTY3
NCwyMTI1NTM4MTk2LDE1ODIzNTAwNTQsLTExOTYyNzM2NTUsNT
Q1Mjk2NTk4LC02NTU5OTM2MDYsLTIxMDMxMjEyOTAsLTExMTQ2
ODU0MDddfQ==
-->