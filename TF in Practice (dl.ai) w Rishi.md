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
<li><a href="#course-1---intro-to-tensorflow">Course 1 - Intro to Tensorflow</a>
<ul>
<li><a href="#week-1">Week 1</a></li>
<li><a href="#week-2">Week 2</a></li>
<li><a href="#week-3">Week 3</a></li>
<li><a href="#week-4">Week 4</a></li>
</ul>
</li>
</ul>

    </div>
  </div>
  <div class="stackedit__right">
    <div class="stackedit__html">
      <h1 id="course-1---intro-to-tensorflow"><a href="https://www.coursera.org/learn/introduction-tensorflow/home/welcome">Course 1 - Intro to Tensorflow</a></h1>
<h2 id="week-1"><a href="https://www.coursera.org/learn/introduction-tensorflow/home/week/1">Week 1</a></h2>
<h3 id="primer">Primer</h3>
<div class="mermaid"><svg xmlns="http://www.w3.org/2000/svg" id="mermaid-svg-Do5LaivblN2XJZm4" width="100%" style="max-width: 391.390625px;" viewBox="0 0 391.390625 147.59375"><g transform="translate(-12, -12)"><g class="output"><g class="clusters"></g><g class="edgePaths"><g class="edgePath" style="opacity: 1;"><path class="path" d="M70.21875,40.3984375L95.21875,40.3984375L152.196894897608,65.3984375" marker-end="url(#arrowhead73)" style="fill:none"></path><defs><marker id="arrowhead73" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g><g class="edgePath" style="opacity: 1;"><path class="path" d="M68.546875,131.1953125L95.21875,131.1953125L152.196894897608,106.1953125" marker-end="url(#arrowhead74)" style="fill:none"></path><defs><marker id="arrowhead74" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g><g class="edgePath" style="opacity: 1;"><path class="path" d="M277.15625,85.796875L302.15625,85.796875L327.15625,85.796875" marker-end="url(#arrowhead75)" style="fill:none"></path><defs><marker id="arrowhead75" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g></g><g class="edgeLabels"><g class="edgeLabel" transform="" style="opacity: 1;"><g transform="translate(0,0)" class="label"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel" transform="" style="opacity: 1;"><g transform="translate(0,0)" class="label"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel" transform="" style="opacity: 1;"><g transform="translate(0,0)" class="label"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span class="edgeLabel"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node" id="Rules" transform="translate(45.109375,40.3984375)" style="opacity: 1;"><rect rx="0" ry="0" x="-25.109375" y="-20.3984375" width="50.21875" height="40.796875"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-15.109375,-10.3984375)"><foreignObject width="30.2249755859375" height="20.80000114440918"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">Rules</div></foreignObject></g></g></g><g class="node" id="A" transform="translate(198.6875,85.796875)" style="opacity: 1;"><rect rx="0" ry="0" x="-78.46875" y="-20.3984375" width="156.9375" height="40.796875"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-68.46875,-10.3984375)"><foreignObject width="136.95001220703125" height="20.80000114440918"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">Traditional Programming</div></foreignObject></g></g></g><g class="node" id="Data" transform="translate(45.109375,131.1953125)" style="opacity: 1;"><rect rx="0" ry="0" x="-23.4375" y="-20.3984375" width="46.875" height="40.796875"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-13.4375,-10.3984375)"><foreignObject width="26.88751220703125" height="20.80000114440918"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">Data</div></foreignObject></g></g></g><g class="node" id="Answers" transform="translate(361.2734375,85.796875)" style="opacity: 1;"><rect rx="0" ry="0" x="-34.1171875" y="-20.3984375" width="68.234375" height="40.796875"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-24.1171875,-10.3984375)"><foreignObject width="48.23748779296875" height="20.80000114440918"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">Answers</div></foreignObject></g></g></g></g></g></g></svg></div>
<h4 id="traditional-programming">Traditional Programming</h4>
<ul>
<li>rules based programming</li>
<li>if-then statements</li>
</ul>
<div class="mermaid"><svg xmlns="http://www.w3.org/2000/svg" id="mermaid-svg-F1tUQVvyaFNdF8li" width="100%" style="max-width: 353.90625px;" viewBox="0 0 353.90625 147.59375"><g transform="translate(-12, -12)"><g class="output"><g class="clusters"></g><g class="edgePaths"><g class="edgePath" style="opacity: 1;"><path class="path" d="M88.234375,40.3984375L113.234375,40.3984375L159.89157685854414,65.3984375" marker-end="url(#arrowhead88)" style="fill:none"></path><defs><marker id="arrowhead88" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g><g class="edgePath" style="opacity: 1;"><path class="path" d="M77.5546875,131.1953125L113.234375,131.1953125L159.89157685854414,106.1953125" marker-end="url(#arrowhead89)" style="fill:none"></path><defs><marker id="arrowhead89" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g><g class="edgePath" style="opacity: 1;"><path class="path" d="M257.6875,85.796875L282.6875,85.796875L307.6875,85.796875" marker-end="url(#arrowhead90)" style="fill:none"></path><defs><marker id="arrowhead90" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g></g><g class="edgeLabels"><g class="edgeLabel" transform="" style="opacity: 1;"><g transform="translate(0,0)" class="label"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel" transform="" style="opacity: 1;"><g transform="translate(0,0)" class="label"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel" transform="" style="opacity: 1;"><g transform="translate(0,0)" class="label"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span class="edgeLabel"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node" id="Answers" transform="translate(54.1171875,40.3984375)" style="opacity: 1;"><rect rx="0" ry="0" x="-34.1171875" y="-20.3984375" width="68.234375" height="40.796875"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-24.1171875,-10.3984375)"><foreignObject width="48.23748779296875" height="20.80000114440918"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">Answers</div></foreignObject></g></g></g><g class="node" id="B" transform="translate(197.9609375,85.796875)" style="opacity: 1;"><rect rx="0" ry="0" x="-59.7265625" y="-20.3984375" width="119.453125" height="40.796875"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-49.7265625,-10.3984375)"><foreignObject width="99.4625244140625" height="20.80000114440918"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">Machine Learning</div></foreignObject></g></g></g><g class="node" id="Data" transform="translate(54.1171875,131.1953125)" style="opacity: 1;"><rect rx="0" ry="0" x="-23.4375" y="-20.3984375" width="46.875" height="40.796875"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-13.4375,-10.3984375)"><foreignObject width="26.88751220703125" height="20.80000114440918"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">Data</div></foreignObject></g></g></g><g class="node" id="Rules" transform="translate(332.796875,85.796875)" style="opacity: 1;"><rect rx="0" ry="0" x="-25.109375" y="-20.3984375" width="50.21875" height="40.796875"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-15.109375,-10.3984375)"><foreignObject width="30.2249755859375" height="20.80000114440918"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">Rules</div></foreignObject></g></g></g></g></g></g></svg></div>
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

    </div>
  </div>
</body>

</html>
