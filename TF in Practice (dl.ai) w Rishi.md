<!DOCTYPE html>
<html>

<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>TF in Practice (dl.ai) w Rishi</title>
  <link rel="stylesheet" href="https://stackedit.io/style.css" />
</head>

<body class="stackedit">
  <div class="stackedit__left">
    <div class="stackedit__toc">
      
<ul>
<li><a href="#specialization---tensorflow-in-practice">Specialization - Tensorflow in Practice</a></li>
<li><a href="#course-1---intro-to-tensorflow">Course 1 - Intro to Tensorflow</a>
<ul>
<li><a href="#week-1">Week 1</a></li>
<li><a href="#week-2">Week 2</a></li>
<li><a href="#week-3">Week 3</a></li>
<li><a href="#week-4">Week 4</a></li>
</ul>
</li>
<li><a href="#course-2---cnn-in-tensorflow">Course 2 - CNN in Tensorflow</a>
<ul>
<li><a href="#week-1-1">Week 1</a></li>
<li><a href="#week-2---image-augmentation">Week 2 - Image Augmentation</a></li>
<li><a href="#week-3---transfer-learning">Week 3 - Transfer Learning</a></li>
<li><a href="#week-4---multiclass-classification">Week 4 - Multiclass Classification</a></li>
</ul>
</li>
</ul>

    </div>
  </div>
  <div class="stackedit__right">
    <div class="stackedit__html">
      <h1 id="specialization---tensorflow-in-practice"><a href="https://www.coursera.org/specializations/tensorflow-in-practice">Specialization - Tensorflow in Practice</a></h1>
<h1 id="course-1---intro-to-tensorflow"><a href="https://www.coursera.org/learn/introduction-tensorflow/home/welcome">Course 1 - Intro to Tensorflow</a></h1>
<h2 id="week-1"><a href="https://www.coursera.org/learn/introduction-tensorflow/home/week/1">Week 1</a></h2>
<h3 id="primer">Primer</h3>
<div class="mermaid"><svg xmlns="http://www.w3.org/2000/svg" id="mermaid-svg-h0bqrsuGjlQsZTaO" width="100%" style="max-width: 445.265625px;" viewBox="0 0 445.265625 158"><g transform="translate(-12, -12)"><g class="output"><g class="clusters"></g><g class="edgePaths"><g class="edgePath" style="opacity: 1;"><path class="path" d="M77.78125,43L102.78125,43L165.59049479166666,68" marker-end="url(#arrowhead91)" style="fill:none"></path><defs><marker id="arrowhead91" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g><g class="edgePath" style="opacity: 1;"><path class="path" d="M75.6953125,139L102.78125,139L165.59049479166666,114" marker-end="url(#arrowhead92)" style="fill:none"></path><defs><marker id="arrowhead92" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g><g class="edgePath" style="opacity: 1;"><path class="path" d="M318.96875,91L343.96875,91L368.96875,91" marker-end="url(#arrowhead93)" style="fill:none"></path><defs><marker id="arrowhead93" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g></g><g class="edgeLabels"><g class="edgeLabel" transform="" style="opacity: 1;"><g transform="translate(0,0)" class="label"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel" transform="" style="opacity: 1;"><g transform="translate(0,0)" class="label"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel" transform="" style="opacity: 1;"><g transform="translate(0,0)" class="label"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span class="edgeLabel"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node" id="Rules" transform="translate(48.890625,43)" style="opacity: 1;"><rect rx="0" ry="0" x="-28.890625" y="-23" width="57.78125" height="46"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-18.890625,-13)"><foreignObject width="37.78125" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">Rules</div></foreignObject></g></g></g><g class="node" id="A" transform="translate(223.375,91)" style="opacity: 1;"><rect rx="0" ry="0" x="-95.59375" y="-23" width="191.1875" height="46"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-85.59375,-13)"><foreignObject width="171.1875" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">Traditional Programming</div></foreignObject></g></g></g><g class="node" id="Data" transform="translate(48.890625,139)" style="opacity: 1;"><rect rx="0" ry="0" x="-26.8046875" y="-23" width="53.609375" height="46"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-16.8046875,-13)"><foreignObject width="33.609375" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">Data</div></foreignObject></g></g></g><g class="node" id="Answers" transform="translate(409.1171875,91)" style="opacity: 1;"><rect rx="0" ry="0" x="-40.1484375" y="-23" width="80.296875" height="46"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-30.1484375,-13)"><foreignObject width="60.296875" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">Answers</div></foreignObject></g></g></g></g></g></g></svg></div>
<h4 id="traditional-programming">Traditional Programming</h4>
<ul>
<li>rules based programming</li>
<li>if-then statements</li>
</ul>
<div class="mermaid"><svg xmlns="http://www.w3.org/2000/svg" id="mermaid-svg-jrZxpJO0B8Z6hDPw" width="100%" style="max-width: 398.40625px;" viewBox="0 0 398.40625 158"><g transform="translate(-12, -12)"><g class="output"><g class="clusters"></g><g class="edgePaths"><g class="edgePath" style="opacity: 1;"><path class="path" d="M100.296875,43L125.296875,43L175.90315755208334,68" marker-end="url(#arrowhead106)" style="fill:none"></path><defs><marker id="arrowhead106" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g><g class="edgePath" style="opacity: 1;"><path class="path" d="M86.953125,139L125.296875,139L175.90315755208334,114" marker-end="url(#arrowhead107)" style="fill:none"></path><defs><marker id="arrowhead107" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g><g class="edgePath" style="opacity: 1;"><path class="path" d="M294.625,91L319.625,91L344.625,91" marker-end="url(#arrowhead108)" style="fill:none"></path><defs><marker id="arrowhead108" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g></g><g class="edgeLabels"><g class="edgeLabel" transform="" style="opacity: 1;"><g transform="translate(0,0)" class="label"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel" transform="" style="opacity: 1;"><g transform="translate(0,0)" class="label"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel" transform="" style="opacity: 1;"><g transform="translate(0,0)" class="label"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span class="edgeLabel"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node" id="Answers" transform="translate(60.1484375,43)" style="opacity: 1;"><rect rx="0" ry="0" x="-40.1484375" y="-23" width="80.296875" height="46"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-30.1484375,-13)"><foreignObject width="60.296875" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">Answers</div></foreignObject></g></g></g><g class="node" id="B" transform="translate(222.4609375,91)" style="opacity: 1;"><rect rx="0" ry="0" x="-72.1640625" y="-23" width="144.328125" height="46"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-62.1640625,-13)"><foreignObject width="124.328125" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">Machine Learning</div></foreignObject></g></g></g><g class="node" id="Data" transform="translate(60.1484375,139)" style="opacity: 1;"><rect rx="0" ry="0" x="-26.8046875" y="-23" width="53.609375" height="46"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-16.8046875,-13)"><foreignObject width="33.609375" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">Data</div></foreignObject></g></g></g><g class="node" id="Rules" transform="translate(373.515625,91)" style="opacity: 1;"><rect rx="0" ry="0" x="-28.890625" y="-23" width="57.78125" height="46"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-18.890625,-13)"><foreignObject width="37.78125" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">Rules</div></foreignObject></g></g></g></g></g></g></svg></div>
<h4 id="machine-learning">Machine Learning</h4>
<ul>
<li>lots of examples + label</li>
<li>rules are inferred by ML</li>
</ul>
<h3 id="hello-world---neural-network">Hello World - Neural Network</h3>
<p><a href="https://github.com/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%202%20-%20Lesson%202%20-%20Notebook.ipynb">colab</a></p>
<blockquote>
<p>X =<br>
Y =<br>
Simple linear curve fitting</p>
</blockquote>
<blockquote>
<p>keras.Sequential()<br>
Define successive layers</p>
</blockquote>
<blockquote>
<p>keras.layers.Dense()<br>
Define a layer of connected neurons<br>
this could be done as a list of Dense() inside Sequential() or Sequential.add(Dense())</p>
</blockquote>
<blockquote>
<p>model.compile(optimizer=‘sgd’, loss=‘mean_squared_error’)<br>
define the loss function, and optimizer - to guess the next value (for gradient descent)</p>
</blockquote>
<blockquote>
<p>model.fit(epochs=N)<br>
actual training of the model</p>
</blockquote>
<blockquote>
<p>model.predict([input])<br>
inference / find predicted values</p>
</blockquote>
<blockquote>
<p>model.evaluate(test_images, test_labels)<br>
evaluate the test set</p>
</blockquote>
<h2 id="week-2"><a href="https://www.coursera.org/learn/introduction-tensorflow/home/week/2">Week 2</a></h2>
<p><a href="https://github.com/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%204%20-%20Lesson%202%20-%20Notebook.ipynb">colab</a></p>
<h3 id="intro-to-computer-vision">Intro to Computer Vision</h3>
<ul>
<li>Help computers look at the images (and understand the content)</li>
<li><a href="https://github.com/zalandoresearch/fashion-mnist">Fashion MNIST</a> -  78k images, 10 categories, 28x28 pixels each</li>
<li>keras has in-build [datasets] (<a href="https://www.tensorflow.org/api_docs/python/tf/keras/datasets">https://www.tensorflow.org/api_docs/python/tf/keras/datasets</a>) including fashion MNIST</li>
</ul>
<blockquote>
<p>fashion_mnist = keras.datasets.fashion_mnist<br>
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()</p>
</blockquote>
<ul>
<li>separated training and test data</li>
<li>labels are enum (numbers) instead of strings - to reduce language <a href="https://developers.google.com/machine-learning/fairness-overview/">bias</a></li>
</ul>
<h3 id="coding-a-cv-neural-network">Coding a CV Neural Network</h3>
<blockquote>
<p>keras.layers.Flatten(input_shape=(28, 28))<br>
Takes the input as a linear array</p>
</blockquote>
<blockquote>
<p>keras.layers.Dense(128, activation=tf.nn.relu)<br>
Hidden Layer</p>
</blockquote>
<blockquote>
<p>keras.layers.Dense(128, activation=tf.nn.softmax)<br>
Output layer</p>
</blockquote>
<h3 id="callback-to-terminate-the-training">Callback to terminate the training</h3>
<pre><code>class myCallback(keras.callbacks.Callback):
def on_epoch_end(self, epoch, logs={}):
    if logs.get('accuracy') &gt; DESIRED_ACCURACY:
    print(f"\nReached {DESIRED_ACCURACY}% accuracy so cancelling training!")
        self.model.stop_training = True

callbacks = myCallback()
model.fit(..., callbacks=[callbacks])
</code></pre>
<h3 id="colab-exporations">Colab Exporations</h3>
<blockquote>
<p>Simple NN (28x28 Flatten, 128 relu, 10 softmax) - no CNN</p>
</blockquote>
<p><a href="https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%204%20-%20Lesson%202%20-%20Notebook.ipynb#scrollTo=rquQqIx4AaGR">Ex 1</a> - What are the numbers?<br>
A - It is the probability that each item is one of the 10 class (and using softmax, we choose the highest probability)</p>
<p><a href="https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%204%20-%20Lesson%202%20-%20Notebook.ipynb#scrollTo=OgQSIfDSOWv6">Ex 2</a> Impact of larger neurons in the hidden layer(512)?<br>
A - More neurons = slower (and more accurate) training; till an extend.</p>
<p><a href="https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%204%20-%20Lesson%202%20-%20Notebook.ipynb#scrollTo=-0lF5MuvSuZF">Ex 5</a> What is the affect of adding additional hidden layer?<br>
A - Usually more helpful but in this particular case, adding an additional layer reduced the accuracy.</p>
<h2 id="week-3"><a href="https://www.coursera.org/learn/introduction-tensorflow/home/week/3">Week 3</a></h2>
<p><a href="https://colab.sandbox.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%206%20-%20Lesson%202%20-%20Notebook.ipynb">colab</a></p>
<h3 id="cnn---convolutions-and-pooling">CNN - Convolutions and Pooling</h3>
<ul>
<li>Convolutions are like filters to extract features from images</li>
<li>Pooling is compressing the result of convolution while reducing the image dimension</li>
</ul>
<blockquote>
<p>training_images=training_images.reshape(60000, 28, 28, 1)<br>
reshape the input</p>
</blockquote>
<blockquote>
<p>tf.keras.layers.Conv2D(num_conv, (3, 3), activation=‘relu’, input_shape=(28, 28, 1))<br>
tk.keras.layers.MaxPooling2D(2, 2)<br>
Stack the Conv2D and MaxPooling2D before the regular NN layers.</p>
</blockquote>
<blockquote>
<p>model.summary()<br>
Shows the NN<br>
the size of image is smaller in the layers because of conv-layers window</p>
</blockquote>
<p><a href="https://lodev.org/cgtutor/filtering.html">Examples of filters</a></p>
<h3 id="colab-exporations-1">Colab Exporations</h3>
<blockquote>
<p>CNN = Conv(64 x (3,3)), MP(2,2), Conv(64 x (3,3)), MP(2,2),  Flatten, Dense(128) + relu, Dense(10) + softmax<br>
Ex 1 - More training might lead to smaller loss with training set but not with validation set (overfitting).</p>
</blockquote>
<h2 id="week-4"><a href="https://www.coursera.org/learn/introduction-tensorflow/home/week/4">Week 4</a></h2>
<h3 id="colabs">Colabs</h3>
<ul>
<li><a href="https://github.com/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%208%20-%20Lesson%202%20-%20Notebook.ipynb">No Validation</a></li>
<li><a href="https://github.com/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%208%20-%20Lesson%203%20-%20Notebook.ipynb">With Validation</a></li>
<li><a href="https://github.com/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%208%20-%20Lesson%204%20-%20Notebook.ipynb">Reduced complexity</a></li>
</ul>
<h3 id="non-uniformcomplex-images">Non-uniform/complex images</h3>
<ul>
<li>different location of subject in the image</li>
<li>much cleaner data</li>
</ul>
<h3 id="imagegenerator-in-tf">ImageGenerator in TF</h3>
<ul>
<li>Uses the directory structure to pick up labels</li>
<li>images need to be of same size (for the TF code)</li>
<li>resized when they are loaded (done by ImageGenerator)</li>
</ul>
<h3 id="convnet-params">ConvNet params</h3>
<ul>
<li>5 layers of CNN 16 -&gt; 32 -&gt; 64 -&gt;64 -&gt; 64</li>
<li>3 channels in inputs (RGB)</li>
<li>Output is single-neuron with sigmoid
<ul>
<li>could have used 2 neurons with softmax</li>
</ul>
</li>
<li>batch_size : to handle multiple input images together</li>
<li>steps_per_epoch : number of steps to train in an epoch
<ul>
<li>save value is num_images / batch_size so that every image is processed once in the epoch</li>
</ul>
</li>
</ul>
<h3 id="convnet-code">ConvNet Code</h3>
<ul>
<li>binary_crossentropy because we are using a binary classifier</li>
<li>RMSProp to specify learning rate</li>
</ul>
<pre><code>from tensorflow.keras.optimizers import RMSprop
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['accuracy'])
</code></pre>
<pre><code># All images will be resized to 150x150
train_datagen = ImageDataGenerator(rescale=1/255)
train_generator = train_datagen.flow_from_directory('/tmp/horse-or-human/', 
target_size=(300, 300), 
batch_size=128, 
class_mode='binary')
</code></pre>
<pre><code>model.fit(train_generator, 
steps_per_epoch=8,
epochs=15,
validation_data=validation_generator,
validation_steps=8,
verbose=2)
</code></pre>
<h3 id="speed-up-at-cost-of-accuracy">Speed up (at cost of accuracy)</h3>
<ul>
<li>by reducing image size</li>
<li>changing number of CNN layers</li>
</ul>
<h1 id="course-2---cnn-in-tensorflow"><a href="https://www.coursera.org/learn/convolutional-neural-networks-tensorflow">Course 2 - CNN in Tensorflow</a></h1>
<h2 id="week-1-1"><a href="https://www.coursera.org/learn/convolutional-neural-networks-tensorflow/home/week/1">Week 1</a></h2>
<h3 id="data-is-coming-from-real-world">Data is coming from real-world</h3>
<ul>
<li>less cleaner</li>
<li>the previous structure works well</li>
</ul>
<pre><code>history = model.fit()
acc = history.history[ 'accuracy' ]
val_acc = history.history[ 'val_accuracy' ]
loss = history.history[ 'loss' ]
val_loss = history.history['val_loss' ]
</code></pre>
<h2 id="week-2---image-augmentation"><a href="https://www.coursera.org/learn/convolutional-neural-networks-tensorflow/home/week/2">Week 2 - Image Augmentation</a></h2>
<h3 id="data-augmentation">Data Augmentation</h3>
<ul>
<li>No additional storage, all transformation are in memory while reading the data</li>
<li>Reduce overfitting</li>
</ul>
<pre><code>train_datagen = ImageDataGenerator(      
	rotation_range=40,      
	width_shift_range=0.2,      
	height_shift_range=0.2,      
	shear_range=0.2,      
	zoom_range=0.2,      
	horizontal_flip=True,      
	fill_mode='nearest')
</code></pre>
<p>From <a href="https://www.tensorflow.org/tensorboard/tensorboard_in_notebooks">Tensorboard in notebook</a></p>
<pre><code>%load_ext tensorboard
import datetime
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
%tensorboard --logdir logs

model.fit(..., callbacks=[tensorboard_callback])
</code></pre>
<h3 id="do-we-need-to-augment-validation-data">Do we need to augment validation data?</h3>
<p><a href="https://www.kaggle.com/c/dogs-vs-cats/overview">https://www.kaggle.com/c/dogs-vs-cats/overview</a></p>
<h2 id="week-3---transfer-learning"><a href="https://www.coursera.org/learn/convolutional-neural-networks-tensorflow/home/week/3">Week 3 - Transfer Learning</a></h2>
<h3 id="transfer-learning">Transfer Learning</h3>
<ul>
<li>Take an existing (trained) model and keep the weights of the top layers fixed(towards CNN) fixed</li>
<li>Enable the lower layers to change</li>
<li>Plug the model in a new model with required output (classification, prediction etc)</li>
<li>Train on new data</li>
</ul>
<p><a href="https://github.com/PracticalDL/Practical-Deep-Learning-Book/blob/master/code/chapter-3/1-keras-custom-classifier-with-transfer-learning.ipynb">Code example</a> from PracticalDL book</p>
<pre><code>from tensorflow.keras.applications.inception_v3 import InceptionV3

local_weights_file = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
pre_trained_model = InceptionV3(input_shape = (150, 150, 3), include_top = False, weights = None)
pre_trained_model.load_weights(local_weights_file)

for layer in pre_trained_model.layers:
	layer.trainable = False

pre_trained_model.summary()

last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output
</code></pre>
<h3 id="training-for-transfer-learning">Training for transfer learning</h3>
<pre><code># Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)

# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)

# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)

# Add a final sigmoid layer for classification
x = layers.Dense(1, activation='sigmoid')(x)

model = Model(pre_trained_model.input, x)
</code></pre>
<h3 id="dropout">Dropout</h3>
<p><a href="https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout">https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout</a></p>
<ul>
<li>Regularization feature</li>
<li>Use for overfitting - mostly for CNN</li>
<li>Cost function (J) is harder to define</li>
</ul>
<h2 id="week-4---multiclass-classification"><a href="https://www.coursera.org/learn/convolutional-neural-networks-tensorflow/home/week/4">Week 4 - Multiclass Classification</a></h2>

<table>
<thead>
<tr>
<th>Code</th>
<th>Binary</th>
<th>Multiclass</th>
</tr>
</thead>
<tbody>
<tr>
<td>train_generator</td>
<td>flow_from_directory(… , class_mode=‘binary’)</td>
<td>flow_from_directory(… , class_mode=‘categorical’)</td>
</tr>
<tr>
<td>final layer</td>
<td>keras.layers.Dense(1, activation=‘sigmoid’)</td>
<td>keras.layers.Dense(num_classes, activation=‘softmax’)</td>
</tr>
<tr>
<td>loss fxn</td>
<td>model.compile(…, loss=‘binary_crossentropy’)</td>
<td>mode.compile(…, loss=‘catagorical_crossentropy’)</td>
</tr>
</tbody>
</table>
    </div>
  </div>
</body>

</html>
