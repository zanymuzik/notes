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

### [Hello World on Neural Network](https://github.com/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%202%20-%20Lesson%202%20-%20Notebook.ipynb)
> X = 
> Y = 
Simple linear curve fitting  

> keras.layers.Dense()
Define a layer of connected neurons

> keras.Sequential()
Define successive layers

> model.compile(optimizer='sgd', loss='mean_squared_error')
define the loss function, and optimizer - to guess the next value (for gradient descent)

> model.fit(epochs=N)
actual training of the model

> model.predict([input])
inference / find predicted values
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTY1NTk5MzYwNiwtMjEwMzEyMTI5MCwtMT
ExNDY4NTQwNywtOTA1NTQ4ODI3LDczMDk5ODExNl19
-->