---


---

<h1 id="specialization---tensorflow-in-practice"><a href="https://www.coursera.org/specializations/tensorflow-in-practice">Specialization - Tensorflow in Practice</a></h1>
<h1 id="course-1---intro-to-tensorflow"><a href="https://www.coursera.org/learn/introduction-tensorflow/home/welcome">Course 1 - Intro to Tensorflow</a></h1>
<h2 id="week-1"><a href="https://www.coursera.org/learn/introduction-tensorflow/home/week/1">Week 1</a></h2>
<h3 id="primer">Primer</h3>
<div class="mermaid"><svg xmlns="http://www.w3.org/2000/svg" id="mermaid-svg-qyh9v9jTOop3LfBn" width="100%" style="max-width: 418.328125px;" viewBox="0 0 418.328125 152.78125762939453"><g transform="translate(-12, -12)"><g class="output"><g class="clusters"></g><g class="edgePaths"><g class="edgePath" style="opacity: 1;"><path class="path" d="M74,41.69531440734863L99,41.69531440734863L158.97992058833273,66.69531440734863" marker-end="url(#arrowhead68)" style="fill:none"></path><defs><marker id="arrowhead68" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g><g class="edgePath" style="opacity: 1;"><path class="path" d="M72.1171875,135.0859432220459L99,135.0859432220459L158.97992058833273,110.0859432220459" marker-end="url(#arrowhead69)" style="fill:none"></path><defs><marker id="arrowhead69" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g><g class="edgePath" style="opacity: 1;"><path class="path" d="M298.0625,88.39062881469727L323.0625,88.39062881469727L348.0625,88.39062881469727" marker-end="url(#arrowhead70)" style="fill:none"></path><defs><marker id="arrowhead70" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g></g><g class="edgeLabels"><g class="edgeLabel" transform="" style="opacity: 1;"><g transform="translate(0,0)" class="label"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel" transform="" style="opacity: 1;"><g transform="translate(0,0)" class="label"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel" transform="" style="opacity: 1;"><g transform="translate(0,0)" class="label"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span class="edgeLabel"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node" id="Rules" transform="translate(47,41.69531440734863)" style="opacity: 1;"><rect rx="0" ry="0" x="-27" y="-21.6953125" width="54" height="43.390625"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-17,-11.6953125)"><foreignObject width="34.00311279296875" height="23.399999618530273"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">Rules</div></foreignObject></g></g></g><g class="node" id="A" transform="translate(211.03125,88.39062881469727)" style="opacity: 1;"><rect rx="0" ry="0" x="-87.03125" y="-21.6953125" width="174.0625" height="43.390625"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-77.03125,-11.6953125)"><foreignObject width="154.0687255859375" height="23.399999618530273"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">Traditional Programming</div></foreignObject></g></g></g><g class="node" id="Data" transform="translate(47,135.0859432220459)" style="opacity: 1;"><rect rx="0" ry="0" x="-25.1171875" y="-21.6953125" width="50.234375" height="43.390625"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-15.1171875,-11.6953125)"><foreignObject width="30.2484130859375" height="23.399999618530273"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">Data</div></foreignObject></g></g></g><g class="node" id="Answers" transform="translate(385.1953125,88.39062881469727)" style="opacity: 1;"><rect rx="0" ry="0" x="-37.1328125" y="-21.6953125" width="74.265625" height="43.390625"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-27.1328125,-11.6953125)"><foreignObject width="54.2672119140625" height="23.399999618530273"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">Answers</div></foreignObject></g></g></g></g></g></g></svg></div>
<h4 id="traditional-programming">Traditional Programming</h4>
<ul>
<li>rules based programming</li>
<li>if-then statements</li>
</ul>
<div class="mermaid"><svg xmlns="http://www.w3.org/2000/svg" id="mermaid-svg-KhCvAElAcC0HlE75" width="100%" style="max-width: 376.15625px;" viewBox="0 0 376.15625 152.78125762939453"><g transform="translate(-12, -12)"><g class="output"><g class="clusters"></g><g class="edgePaths"><g class="edgePath" style="opacity: 1;"><path class="path" d="M94.265625,41.69531440734863L119.265625,41.69531440734863L167.95643780117024,66.69531440734863" marker-end="url(#arrowhead83)" style="fill:none"></path><defs><marker id="arrowhead83" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g><g class="edgePath" style="opacity: 1;"><path class="path" d="M82.25,135.0859432220459L119.265625,135.0859432220459L167.95643780117024,110.0859432220459" marker-end="url(#arrowhead84)" style="fill:none"></path><defs><marker id="arrowhead84" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g><g class="edgePath" style="opacity: 1;"><path class="path" d="M276.15625,88.39062881469727L301.15625,88.39062881469727L326.15625,88.39062881469727" marker-end="url(#arrowhead85)" style="fill:none"></path><defs><marker id="arrowhead85" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g></g><g class="edgeLabels"><g class="edgeLabel" transform="" style="opacity: 1;"><g transform="translate(0,0)" class="label"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel" transform="" style="opacity: 1;"><g transform="translate(0,0)" class="label"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel" transform="" style="opacity: 1;"><g transform="translate(0,0)" class="label"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span class="edgeLabel"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node" id="Answers" transform="translate(57.1328125,41.69531440734863)" style="opacity: 1;"><rect rx="0" ry="0" x="-37.1328125" y="-21.6953125" width="74.265625" height="43.390625"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-27.1328125,-11.6953125)"><foreignObject width="54.2672119140625" height="23.399999618530273"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">Answers</div></foreignObject></g></g></g><g class="node" id="B" transform="translate(210.2109375,88.39062881469727)" style="opacity: 1;"><rect rx="0" ry="0" x="-65.9453125" y="-21.6953125" width="131.890625" height="43.390625"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-55.9453125,-11.6953125)"><foreignObject width="111.89532470703125" height="23.399999618530273"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">Machine Learning</div></foreignObject></g></g></g><g class="node" id="Data" transform="translate(57.1328125,135.0859432220459)" style="opacity: 1;"><rect rx="0" ry="0" x="-25.1171875" y="-21.6953125" width="50.234375" height="43.390625"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-15.1171875,-11.6953125)"><foreignObject width="30.2484130859375" height="23.399999618530273"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">Data</div></foreignObject></g></g></g><g class="node" id="Rules" transform="translate(353.15625,88.39062881469727)" style="opacity: 1;"><rect rx="0" ry="0" x="-27" y="-21.6953125" width="54" height="43.390625"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-17,-11.6953125)"><foreignObject width="34.00311279296875" height="23.399999618530273"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">Rules</div></foreignObject></g></g></g></g></g></g></svg></div>
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
</table><h3 id="ai-need-to-research-">AI: Need to research-</h3>
<p>1 <a href="https://keras.io/api/losses/">https://keras.io/api/lKeras::Losses/</a><br>
2. Try with one-hot encoding</p>
<p>sparse_categorical_crossentropy vs categorical_crossentropy<br>
<a href="https://jovianlin.io/cat-crossentropy-vs-sparse-cat-crossentropy/">https://jovianlin.io/cat-crossentropy-vs-sparse-cat-crossentropy/</a></p>
<p>1-hot encoding vs labels directly</p>
<h1 id="course-3---nlp-in-tensorflow"><a href="https://www.coursera.org/learn/natural-language-processing-tensorflow">Course 3 - NLP in Tensorflow</a></h1>
<h2 id="week-1---tokenization"><a href="https://www.coursera.org/learn/natural-language-processing-tensorflow/home/week/1">Week 1 - Tokenization</a></h2>
<p><a href="https://github.com/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%201%20-%20Lesson%203.ipynb">Week 1 Notebook</a></p>
<h3 id="word-based-encodings">Word Based Encodings</h3>
<ul>
<li>Idea1: Use ASCII values for characters
<ul>
<li>same characters (in different order) could mean different things</li>
<li>ex. LISTEN vs SILENT</li>
</ul>
</li>
<li>Idea2: Tokenizing at word level
<ul>
<li>tf and keras have in-build API - tokenizer</li>
<li>strips punctuations and lowercases automatically</li>
<li>builds a dictionary of tokens</li>
</ul>
</li>
</ul>
<pre><code>from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words = 100)
tokenizer.fit_on_texts(sentences)
lexicon = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)
</code></pre>
<h3 id="preprocessing">Preprocessing</h3>
<h4 id="missing-words">Missing words</h4>
<ul>
<li>Handled by an OOV token</li>
</ul>
<pre><code>tokenizer = Tokenizer(num_words = 100, oov_token="&lt;OOV&gt;")
</code></pre>
<h4 id="padding">Padding</h4>
<ul>
<li>change sentence length to same by pre/post padding with zeros</li>
</ul>
<pre><code>from tensorflow.keras.preprocessing.sequence import pad_sequences
padded = pad_sequences(test_seq, maxlen=10, padding='post')
</code></pre>
<h2 id="week-2---embeddings"><a href="https://www.coursera.org/learn/natural-language-processing-tensorflow/home/week/2">Week 2 - Embeddings</a></h2>
<pre><code>model = tf.keras.Sequential([
tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(6, activation='relu'),
tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
</code></pre>
<p>Embeddings</p>
<ul>
<li>Another level after tokenization</li>
<li>Meaning of words in n-dim space</li>
</ul>
<h3 id="build-in-datasets-in-tf">Build-in datasets in TF</h3>
<ul>
<li>IMDB dataset for sentiment analysis</li>
<li><a href="https://www.tensorflow.org/datasets/catalog/overview">https://www.tensorflow.org/datasets/catalog/overview</a></li>
<li><a href="https://github.com/tensorflow/datasets/tree/master/docs/catalog">https://github.com/tensorflow/datasets/tree/master/docs/catalog</a></li>
</ul>
<pre><code>import tensorflow_datasets as tfds
imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
</code></pre>
<h3 id="code-pointers">Code pointers</h3>
<ul>
<li>Tokenize both datasets and feed in the NN</li>
<li>Lexicon is based on training set so expect more OOV in test set</li>
<li>Words are represented as vectors - that are input to the NN</li>
<li>NN learns to associate the vectors with labels based on the training</li>
<li>Embedding layer in the beginning</li>
<li>Result of embedding layer is matrix [length of words x num-dim of embeddings]</li>
<li>Flatten or GlobalAveragePooling1D to create single dim vector</li>
<li>Flatten is slower and more accurate</li>
<li>train and visualize the embedding vectors on <a href="http://projector.tensorflow.org">http://projector.tensorflow.org</a></li>
</ul>
<pre><code>model = tf.keras.Sequential([
tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(6, activation='relu'),
tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

num_epochs = 10
model.fit(padded, training_labels_final, epochs=num_epochs, validation_data=(testing_padded, testing_labels_final))
</code></pre>
<h3 id="subwordstextencoder">SubwordsTextEncoder</h3>
<ul>
<li><a href="https://www.tensorflow.org/datasets/api_docs/python/tfds/features/text/SubwordTextEncoder">SubwordTextEncoder</a></li>
</ul>
<pre><code>import tensorflow_datasets as tfds
imdb, info = tfds.load("imdb_reviews/subwords8k", with_info=True, as_supervised=True)
tokenizer = info.features['text'].encoder

tokenized_string = tokenizer.encode(sample_string)
print ('Tokenized string is {}'.format(tokenized_string))

original_string = tokenizer.decode(tokenized_string)
print ('The original string: {}'.format(original_string))
</code></pre>
<h3 id="ai---need-to-research">AI - need to research</h3>
<ul>
<li><a href="https://keras.io/api/layers/pooling_layers/">keras::PoolingLayer</a></li>
<li>GlobalAveragePooling1D vs Flatten</li>
<li>Q - Is any forcing function sufficient (ratings, sarcasm etc) to generate the embeddings? how are the embeddings affected by the forcing fxn?</li>
</ul>
<h2 id="week-3---sequence-models"><a href="https://www.coursera.org/learn/natural-language-processing-tensorflow/home/week/3">Week 3 - Sequence Models</a></h2>
<h3 id="rnn---lstm">RNN - LSTM</h3>
<ul>
<li>Context is important - and it could be much earlier in a sentence</li>
</ul>
<h4 id="single-lstm">Single LSTM</h4>
<pre><code>model = tf.keras.Sequential([
tf.keras.layers.Embedding(tokenizer.vocab_size, 64),
tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
tf.keras.layers.Dense(64, activation='relu'),
tf.keras.layers.Dense(1, activation='sigmoid')
])
</code></pre>
<h4 id="dual-lstm">Dual LSTM</h4>
<pre><code>model = tf.keras.Sequential([
tf.keras.layers.Embedding(tokenizer.vocab_size, 64),
tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
tf.keras.layers.Dense(64, activation='relu'),
tf.keras.layers.Dense(1, activation='sigmoid')
])
</code></pre>
<h4 id="conv1d">Conv1D</h4>
<pre><code>model = tf.keras.Sequential([
tf.keras.layers.Embedding(tokenizer.vocab_size, 64),
tf.keras.layers.Conv1D(128, 5, activation='relu'),
tf.keras.layers.GlobalAveragePooling1D(),
tf.keras.layers.Dense(64, activation='relu'),
tf.keras.layers.Dense(1, activation='sigmoid')
])
</code></pre>
<h4 id="ai---return-and-finish-this-week">AI - Return and finish this week</h4>
<h2 id="week-4---text-synthesis"><a href="https://www.coursera.org/learn/natural-language-processing-tensorflow/home/week/4">Week 4 - Text Synthesis</a></h2>
<h3 id="text-generation">Text Generation</h3>
<ul>
<li>Actually a prediction problem</li>
<li>Next word = y, previous word = x (and use a DNN)</li>
<li>Using Embedding in the input and 1-hot in the output</li>
</ul>
<h3 id="example">Example</h3>
<ul>
<li>Tokenize a paragraph of text</li>
<li>For bigger corpus, it makes sense to use character-level generation (to reduce the size in one-hot encoding y)</li>
<li><a href="https://www.tensorflow.org/tutorials/text/text_generation">https://www.tensorflow.org/tutorials/text/text_generation</a></li>
<li><a href="https://stackabuse.com/text-generation-with-python-and-tensorflow-keras/">https://stackabuse.com/text-generation-with-python-and-tensorflow-keras/</a>](<a href="https://stackabuse.com/text-generation-with-python-and-tensorflow-keras/">https://stackabuse.com/text-generation-with-python-and-tensorflow-keras/</a>)</li>
</ul>
<h3 id="questions">Questions</h3>
<ul>
<li>Why do we 1-hot encode only the y?</li>
<li>Ans - no, in this case, they used the embedding layer for the x (input). In general, 1-hot is used for both. (insert keras example)</li>
</ul>
<h1 id="course-4---seq-ts-prediction"><a href="https://www.coursera.org/learn/tensorflow-sequences-time-series-and-prediction/">Course 4 - Seq, TS, Prediction</a></h1>
<h2 id="week-1---time-series"><a href="https://www.coursera.org/learn/tensorflow-sequences-time-series-and-prediction/home/week/1">Week 1 - Time Series</a></h2>
<h3 id="time-series">Time Series</h3>
<ul>
<li>Stock Markets, Weather, Moore’s law etc</li>
<li>Single value per time step = univariate</li>
<li>Multiple values per time step = multivariate</li>
</ul>
<h4 id="anything-that-has-a-time-factor">Anything that has a time factor</h4>
<ul>
<li>ML helps with
<ul>
<li>forecasting</li>
<li>imputation (looking back and filling in data)</li>
<li>filling holes from missing data</li>
<li>detecting DOS</li>
<li>split sequence - for example, sound</li>
</ul>
</li>
</ul>
<h4 id="common-patterns-in-ts">Common Patterns in TS</h4>
<ul>
<li>Trends - specific direction</li>
<li>Seasonality - particular intervals
<ul>
<li>Sometimes Trends + Seasonality combo</li>
</ul>
</li>
<li>Auto-correlated (correlates with delayed copy of itself = lag)
<ul>
<li>TS with memory</li>
<li>Unpredictable spikes = innovations</li>
</ul>
</li>
<li>White noise (no correlation)</li>
<li>Real time TS are mix of all 4</li>
</ul>
<h4 id="non-stationary-ts">Non-stationary TS</h4>
<ul>
<li>Big events can change the TS characters</li>
<li>Better to use a time window and be specific in time while training</li>
</ul>
<h3 id="forecasting-techniques">Forecasting Techniques</h3>
<h4 id="fixed-partitioning">Fixed Partitioning</h4>
<ul>
<li>Validating based on (training, validation and test periods)</li>
<li>Train model and test and then retrain using the test data (to capture the latest data)</li>
</ul>
<h3 id="metrics">Metrics</h3>
<ul>
<li>Error = forecast - predicted</li>
<li>MSE = Means Square Error  =
<ul>
<li>np.square(error).mean()</li>
</ul>
</li>
<li>RMSE = Root Mean Square Error =
<ul>
<li>np.sqrt(mse)</li>
</ul>
</li>
<li>MAE = Mean Absolute Error =
<ul>
<li>np.abs(error).mean()</li>
</ul>
</li>
<li>MAPE = Mean Absolute Percentage Error
<ul>
<li>np.abs(errors / x_valid).mean()</li>
</ul>
</li>
</ul>
<h3 id="types-of-forecasts">Types of Forecasts</h3>
<h4 id="naive-forecasting">Naive forecasting</h4>
<ul>
<li>next value = last value</li>
<li>Trend + seasonality + noise</li>
</ul>
<h4 id="moving-average">Moving Average</h4>
<ul>
<li>Moving window of average from past values</li>
<li>Removes the noise</li>
<li>But doesn’t capture trend and seasonality</li>
</ul>
<h4 id="differencing">Differencing</h4>
<ul>
<li>NewSeries(t) = Series(t) - Series(t - T), sayT = 365</li>
<li>Removes seasonality and trends</li>
<li>Now use moving average on the new series to predict</li>
<li>And add back Series(t - T)</li>
<li>This keeps the noise because of past values
<ul>
<li>which could be fixed by another moving average over those values</li>
</ul>
</li>
</ul>
<h4 id="exercises">Exercises</h4>
<ul>
<li><a href="https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%204%20-%20S%2BP/S%2BP_Week_1_Lesson_2.ipynb">Week1-Lesson2</a></li>
<li><a href="https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%204%20-%20S%2BP/S%2BP%20Week%201%20-%20Lesson%203%20-%20Notebook.ipynb#scrollTo=y7QztBIVR1tb">Week1-Lesson3</a></li>
</ul>
<p>Q: Does the moving average of Series(t-T) has a different window size?<br>
Q: What the hell is a centered window? (<a href="https://www.coursera.org/learn/tensorflow-sequences-time-series-and-prediction/lecture/4s0E8/trailing-versus-centered-windows">https://www.coursera.org/learn/tensorflow-sequences-time-series-and-prediction/lecture/4s0E8/trailing-versus-centered-windows</a>)</p>
<h2 id="week-2---ml-for-time-series"><a href="https://www.coursera.org/learn/tensorflow-sequences-time-series-and-prediction/home/week/2">Week 2 - ML for Time Series</a></h2>
<h3 id="ai---redo-this-week-if-possible">AI - redo this week if possible</h3>
<h3 id="features-and-labels">Features and Labels</h3>
<ul>
<li>Features = Number of values</li>
<li>Labels = Next value</li>
</ul>
<h4 id="colab-1">Colab 1</h4>
<ul>
<li>Week 2 - Lesson 1](<a href="https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%204%20-%20S%2BP/S%2BP%20Week%202%20Lesson%201.ipynb">https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow In Practice/Course 4 - S%2BP/S%2BP Week 2 Lesson 1.ipynb</a>)</li>
</ul>
<pre><code>dataset = tf.data.Dataset.range(10)
for val in dataset:
	print(val.numpy())
</code></pre>
<ul>
<li>Window</li>
</ul>
<pre><code>dataset = dataset.window(5, shift=1)
for window_dataset in dataset:
	for val in window_dataset:
		print(val.numpy(), end=" ")
	print()
</code></pre>
<p>[missing]</p>
<ul>
<li>Final code</li>
</ul>
<pre><code>dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
dataset = dataset.map(lambda window:  (window[:-1], window[-1:]))
dataset = dataset.shuffle(buffer_size=10)
dataset = dataset.batch(2).prefetch(1)
for x,y in dataset:
	print("x = ", x.numpy())
	print("y = ", y.numpy())
</code></pre>
<ul>
<li>Sequence Bias<br>
Order of the input can mess the selection</li>
</ul>
<h4 id="windowed-dataset-into-neural-network">Windowed dataset into Neural Network</h4>
<ul>
<li>Use of a shuffle buffer</li>
</ul>
<pre><code>def  windowed_dataset(series, window_size, batch_size, shuffle_buffer):
	dataset = tf.data.Dataset.from_tensor_slices(series)
	dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
	dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
	dataset = dataset.shuffle(shuffle_buffer).map(lambda window:  (window[:-1], window[-1]))
	dataset = dataset.batch(batch_size).prefetch(1)
	return dataset
</code></pre>
<h4 id="single-layer-neural-network">Single layer Neural Network</h4>
<h2 id="week-3---rnn-for-time-series"><a href="https://www.coursera.org/learn/tensorflow-sequences-time-series-and-prediction/home/week/3">Week 3 - RNN for Time Series</a></h2>
<h3 id="rnn-for-prediction">RNN for prediction</h3>
<div class="mermaid"><svg xmlns="http://www.w3.org/2000/svg" id="mermaid-svg-D1oJpu9jFJ7ulKod" width="100%" style="max-width: 106.3125px;" viewBox="0 0 106.3125 459.76564025878906"><g transform="translate(-12, -12)"><g class="output"><g class="clusters"></g><g class="edgePaths"><g class="edgePath" style="opacity: 1;"><path class="path" d="M65.15625,113.39062881469727L65.15625,88.39062881469727L65.15625,63.390628814697266" marker-end="url(#arrowhead107)" style="fill:none"></path><defs><marker id="arrowhead107" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g><g class="edgePath" style="opacity: 1;"><path class="path" d="M65.15625,206.78125762939453L65.15625,181.78125762939453L65.15625,156.78125762939453" marker-end="url(#arrowhead108)" style="fill:none"></path><defs><marker id="arrowhead108" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g><g class="edgePath" style="opacity: 1;"><path class="path" d="M65.15625,300.1718864440918L65.15625,275.1718864440918L65.15625,250.1718864440918" marker-end="url(#arrowhead109)" style="fill:none"></path><defs><marker id="arrowhead109" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g><g class="edgePath" style="opacity: 1;"><path class="path" d="M65.15625,393.56251525878906L65.15625,368.56251525878906L65.15625,343.56251525878906" marker-end="url(#arrowhead110)" style="fill:none"></path><defs><marker id="arrowhead110" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g></g><g class="edgeLabels"><g class="edgeLabel" transform="" style="opacity: 1;"><g transform="translate(0,0)" class="label"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel" transform="" style="opacity: 1;"><g transform="translate(0,0)" class="label"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel" transform="" style="opacity: 1;"><g transform="translate(0,0)" class="label"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel" transform="" style="opacity: 1;"><g transform="translate(0,0)" class="label"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span class="edgeLabel"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node" id="Dense" transform="translate(65.15625,135.0859432220459)" style="opacity: 1;"><rect rx="0" ry="0" x="-30.1640625" y="-21.6953125" width="60.328125" height="43.390625"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-20.1640625,-11.6953125)"><foreignObject width="40.33123779296875" height="23.399999618530273"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">Dense</div></foreignObject></g></g></g><g class="node" id="forecasts" transform="translate(65.15625,41.69531440734863)" style="opacity: 1;"><rect rx="0" ry="0" x="-38.59375" y="-21.6953125" width="77.1875" height="43.390625"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-28.59375,-11.6953125)"><foreignObject width="57.19219970703125" height="23.399999618530273"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">forecasts</div></foreignObject></g></g></g><g class="node" id="Recurrent2" transform="translate(65.15625,228.47657203674316)" style="opacity: 1;"><rect rx="0" ry="0" x="-45.15625" y="-21.6953125" width="90.3125" height="43.390625"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-35.15625,-11.6953125)"><foreignObject width="70.3125" height="23.399999618530273"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">Recurrent2</div></foreignObject></g></g></g><g class="node" id="Recurrent1" transform="translate(65.15625,321.86720085144043)" style="opacity: 1;"><rect rx="0" ry="0" x="-45.15625" y="-21.6953125" width="90.3125" height="43.390625"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-35.15625,-11.6953125)"><foreignObject width="70.3125" height="23.399999618530273"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">Recurrent1</div></foreignObject></g></g></g><g class="node" id="TSData" transform="translate(65.15625,428.66407775878906)" style="opacity: 1;"><circle x="-35.1015625" y="-21.6953125" r="35.1015625"></circle><g class="label" transform="translate(0,0)"><g transform="translate(-25.1015625,-11.6953125)"><foreignObject width="50.203125" height="23.399999618530273"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">TS Data</div></foreignObject></g></g></g></g></g></g></svg></div>
<h3 id="shape-of-the-data">Shape of the data</h3>
<ul>
<li>Input =  Batch size x Window size x data-dim (1 for univariate)</li>
<li>Output = Batch size x  Window size x Memory size (3 in this example)
<ul>
<li>Add explanation as explained to Rishi on the phone</li>
</ul>
</li>
<li>Hi = Yi (in simple RNN)</li>
<li>set input_shape = [None] to take any size (for the data-dim)
<ul>
<li>so we can use the same code for any number of layers for the input data</li>
</ul>
</li>
<li>Sequence to vector RNN
<ul>
<li>the intermediate outputs (Y0 to YN-1) are not really important</li>
<li>ignore all output except the last one</li>
<li>controlled by return_sequences = True</li>
</ul>
</li>
<li>If this is the intermediate RNN, then we do need the intermediate layer (for the higher level RNN)
<ul>
<li>last RNN layer typically has return_sequences = False</li>
</ul>
</li>
</ul>
<h3 id="lambda-layers">lambda layers</h3>
<pre><code># help with dimentions (to go from univariate data -&gt; vector)
keras.layers.Lambda(lambda x: tf.expand_dims(x, axis = -1), input_shape = [None])

# scaling inputs
keras.layers.Lambda(lambda x: x * 100.0)

# Tune the learning rate
tf.keras.callback.LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch/20)) 
</code></pre>
<h3 id="estimate-learning-rate">Estimate learning rate</h3>
<ul>
<li><a href="https://en.wikipedia.org/wiki/Huber_loss">Huber Loss</a></li>
<li>AI: understand how the learning rate schedule works. It is not intuitive to have a plot of LR wrt loss to estimate the best LR for the network.</li>
</ul>
<h3 id="lstm">LSTM</h3>
<ul>
<li>some magic with state (which is missing in RNN)</li>
<li>state could be bi-directional</li>
</ul>
<pre><code>model = tf.keras.models.Sequential([
tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),input_shape=[None]),
tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)), 
tf.keras.layers.Dense(1),
tf.keras.layers.Lambda(lambda x: x * 100.0)
])
</code></pre>
<h2 id="week-4---cnn-for-time-series"><a href="https://www.coursera.org/learn/tensorflow-sequences-time-series-and-prediction/home/week/4">Week 4 - CNN for Time Series</a></h2>
<ul>
<li>Real World Data</li>
<li>Stack CNN layer with LSTM</li>
<li>1 Dim CNN window (input_shape = [None, 1])</li>
</ul>

