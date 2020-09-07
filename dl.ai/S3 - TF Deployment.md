# Specialization TF Deployment

# Course 1 - Browser Based Models with TF.js
## [Javascript in TensorFlow](https://www.tensorflow.org/js)

## [Week 1 - simple NN in js](https://www.coursera.org/learn/browser-based-models-tensorflow/home/week/1)

[Github Repo](https://github.com/lmoroney/dlaicourse/tree/master/TensorFlow%20Deployment)
- tensorflow.js supports layers api from keras
- lower api is core.api
- core can run on node.js too
- tensor2d is used (numpy is not available in js)
- training is asynchronous and based on callbacks because we don't want to lock the browser


### Simple Model Training - change me

![](https://miro.medium.com/max/1400/0*oY2OG7MFBN4eK1AN.)

[FirstHTML.html](https://github.com/lmoroney/dlaicourse/blob/master/TensorFlow%20Deployment/Course%201%20-%20TensorFlow-JS/Week%201/Examples/FirstHTML.html)

```
<script lang="js>
	async function doTraining(model){
		const history = await model.fit(xs, ys,
			{ epochs: 500,
			  callbacks:{onEpochEnd: async(epoch, logs) => {
				console.log("Epoch:" + epoch + " Loss:" + logs.loss);
			  }
			}
		});
	}

	const model = tf.sequencial();
	model.add(tf.laters.dense({units: 1, inputShape: [1]}));
	model.compile({loss:'meanSquaredError', optimizer:'sgd'});
	model.summary();
	  
	const xs = tf.tensor2d([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], [6, 1]);
	const ys = tf.tensor2d([-3.0, -1.0, 2.0, 3.0, 5.0, 7.0], [6, 1]);
	doTraining(model).then(() => {
		alert(model.predict(tf.tensor2d([10], [1,1])));
</script>
```
- training using an async function - that calls back once done (because training could take forever)
- inference data is passed as a tensor2d

### Training Models with CSV
[iris-classifier.html](https://github.com/lmoroney/dlaicourse/blob/master/TensorFlow%20Deployment/Course%201%20-%20TensorFlow-JS/Week%201/Examples/iris-classifier.html)

- tf.data.csv() to load csv data
- cvs file is loaded from the same place as html (web server)
- first line of cvs (headers) need to have labels
- label is marked as-
```
columnConfigs: {
	species: {
		isLabel: true
	}
}
```
- one hot-encoding using map (over the species column)
```
trainingData.map(({xs, ys}) => {
	const labels = [
		ys.species == "setosa" ? 1 : 0,
		ys.species == "virginica" ? 1 : 0,
		ys.species == "versicolor" ? 1 : 0
	]
return{ xs: Object.values(xs), ys: Object.values(labels)};
```
- xs are not mapped (array of array) + ys (mapped)

## incomplete

## [Week 2 - CNN in js](https://www.coursera.org/learn/browser-based-models-tensorflow/home/week/2)

### CNN Training in browser
[code](https://github.com/lmoroney/dlaicourse/blob/master/TensorFlow%20Deployment/Course%201%20-%20TensorFlow-JS/Week%202/Examples/script.js)

#### Defining the CNN model
```
function getModel() {
	model = tf.sequential();
	model.add(tf.layers.conv2d({inputShape: [28, 28, 1], kernelSize: 3, 
								filters: 8, activation: 'relu'}));
	model.add(tf.layers.maxPooling2d({poolSize: [2, 2]}));
	model.add(tf.layers.conv2d({filters: 16, kernelSize: 3, activation: 'relu'}));
	model.add(tf.layers.maxPooling2d({poolSize: [2, 2]}));
	model.add(tf.layers.flatten());
	model.add(tf.layers.dense({units: 128, activation: 'relu'}));
	model.add(tf.layers.dense({units: 10, activation: 'softmax'}));
	model.compile({ optimizer: tf.train.adam(), 
					loss: 'categoricalCrossentropy', 
					metrics: ['accuracy']});
	return model;
}
```

#### Training the model
```
const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

return model.fit(trainXs, trainYs, {
	batchSize: BATCH_SIZE,
	validationData: [testXs, testYs],
	epochs: 20,
	shuffle: true,
	callbacks: fitCallbacks
});
```
- fitCallbacks are provided by [tfjs-vis](https://github.com/tensorflow/tfjs/tree/master/tfjs-vis) library for visualizations

#### Sprite Sheets
- prohibitively expensive to load individual images (from a web browser)
- load them all together as a sprite sheet ex - [MNIST sprite sheet.png](https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png)

## Course Incomplete

# [Course 2 - Device based Models with TFLite](https://www.coursera.org/learn/device-based-models-tensorflow)

[https://www.tensorflow.org/lite](https://www.tensorflow.org/lite)

## [Week 1 - Intro](https://www.coursera.org/learn/device-based-models-tensorflow/home/week/1)

### Mobile AI Platform
- Lightweight
- Low latency device
- Avoid round trip to server (to make a ML RPC call)
	- Also more privacy sensitive
- Low Power
- Pre trained model

### Components of TFLite
#### Converter - outside the device
- create tflite models
- reduce size
#### Interpreter - on the device
- inference of converter models
- uses reduce set of tf ops

### Architecture
- Common format for mobile device
- Inference performance on the device is important
	- software - use NN API on android devices
	- hardware - use edge TPU, GPU or CPU optimizations
- [GPU Delegates](https://youtu.be/QSbAUxWfxQw)
	- graph execution happens on hardware, using device GPU
	- Optimize ops based on coalesces and re-writes

### Techniques
#### Quantization
- Reducing model precision
#### Weight Pruning
- Reducing the overall number of parameters
#### Model Topology Transform
- Changes the model topology to make model more efficient 

#### Why quantization
- All CPU platforms are supported
- Reduce latency and inference costs
- Allow execution on hardware restricted to fixed point ops
- Optimize models for special purpose HW accelerators (TPUs)

### Save Model Format
- TF Model -> Saved Model -> | Convertor | -> tflite model
```
tf.lite.TFLiteConverter.from_saved_model() or
tf.lite.TFLiteConverter.from_keras_model() or
tf.lite.TFLiteConverter.from_concrete_functions()
```
- SavedModel is universal API for saving a TF model
	- Metagraph for metadata in the model
	- Allows versioning of models

### Code

#### From SavedModel
```
# Export the saved model
export_dir = '/tmp/saved_model'
tf.saved_model.save(model, export_dir)

# Convert
converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
tflite_model = converter.convert()

# Save the tflite model
tflite_model_file = pathlib.Path('/tmp/foo.tflite')
tflite_model_file.write_bytes(tflite_model)
```

#### From Keras (Pre-existing model)
```
# Load the Mobilenet tf.keras model
model = tf.keras.applications.MobileNetV2(weights="imagenet", input_shape=(224, 224, 3))
model.save('model.h5')

# Convert
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the tflite model
tflite_model_file = pathlib.Path('/tmp/foo.tflite')
tflite_model_file.write_bytes(tflite_model)
```

#### Command line usage
```
# From SavedModel
tflite_convert --output_file=model.tflite --saved_model_file=/tmp/saved_model

# From Keras
tflite_convert --output_file=model.tflite --keras_model_file=model.h5
```

### Post training Quantization
- Quantization after training is done on the model
	- vs quantizing the model during training
- Improvement in model size and latency
	- at the cost of degradation in model accuracy 
- OPTIMIZE_FOR_SIZE or OPTIMIZE_FOR_LATENCY
	- or let it choose the right balance (DEFAULT)
- For Edge TPUs, accelerator uses only integers for Edge TPUs
	- quantization can reduce size by 4x
- Use generator 
	- use a representation dataset to optimize the tflite model
	- COME BACK AND UNDERSTAND

### TF-Select - experimental feature
- Not all models are supported for conversion 
	- tflite op set is smaller than regular set
- TF-Select to overcome unsupported ops
```
# Convert
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
			tf.lite.OpsSet.SELECT_TF_OPS]			 
tflite_model = converter.convert()
```

### TFLite Interpreter in Python 
- Does not need the mobile device to test
	- could be tested on servers/desktops
```
# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Run the interpreter
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
tflite_results = interpreter.get_tensor(output_details[0]['index'])

```
### Running the model
Continue from [https://www.coursera.org/learn/device-based-models-tensorflow/lecture/R3tiz/running-the-models](https://www.coursera.org/learn/device-based-models-tensorflow/lecture/R3tiz/running-the-models)

### Transfer Learning
## INCOMPLETE



## [Week 2 - Android](https://www.coursera.org/learn/device-based-models-tensorflow/home/week/2)

### App Architecture
```mermaid
graph LR;
A(View) 
B(Preprocess Image) 
C(TFLite Classifier) 
D(Result)
A --> B 
B --> C 
C --> D
```

### Interpreter
#### Initialize the interpreter
```
val i = Interpreter.Options()
i.setNumThreads()
i.setUseNNAPI(true) # Enable hw acceleration using NN API
i.setAllowFp16PrecisionForFp32() # use less precision
i.addDelegate(true) # for GPU
```

#### Load the model
```
assetManager.openFd("model")
tflite = Interpreter(model, options)
```

#### Prepare Input
- input needs to be resized
- ARGB values -> normalized RGB for model inference
- individual color (R,G,B) = shift and mask 
- normalize for each channel
```
// to resize
Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, false)

// create bytebuffer
```
#### Inference
```
val result = Array(1) { FloatArray(2) }
interpreter.run(byteBuffer, result)

# use an android priority queue to sort the result
# and take top value (based on confidence)
```
## INCOMPLETE







## [Week 3 - iOS ](https://www.coursera.org/learn/device-based-models-tensorflow/home/week/3)

## skipped

## [Week 4 - Raspberry Pi ](https://www.coursera.org/learn/device-based-models-tensorflow/home/week/4)

### Adoption of Smart devices > Mobile phones
- Smaller, faster model
- On device accelerators
- Local inference for disconnected devices
	- avoid RTT latency
- Privacy is better
- Better power consumption (without network connection)

###  Coral devices
- [Edge TPU](https://cloud.google.com/edge-tpu) (using dev board or accelerator)
- MendelOS (debian based) + toolkit (mdt)
- Edge TPU Compiler 
- Edge TPU [Models](https://coral.ai/models/)
- [Coral github](https://github.com/google-coral)

### RaspPi
- Raspbian OS - debian based
- Can be clubbed with coral accelerator for inference

### Micro-controller
- Can be embedded into hardware
- Reduced memory/storage
- small form factor
- designed to work offline
- ex - [Sparkfun Edge](https://www.sparkfun.com/products/15170)

### TF on single board 
 1. Compile TF from source -> creates whl file
 2. Install with pip
 3. Use TF interpreter only (tflite_runtime)

### [Image Classification](https://www.tensorflow.org/lite/models/image_classification/overview)
- Pre-quantized Mobilenet trained on ImageNet
- 1000 classes
- [classify.py](https://github.com/lmoroney/dlaicourse/blob/master/TensorFlow%20Deployment/Course%202%20-%20TensorFlow%20Lite/Week%204/Examples/image_classification/classify.py)  code is different than classification.py (video) - figure [why](https://www.coursera.org/learn/device-based-models-tensorflow/discussions/weeks/4/threads/Unq9EI7HS4S6vRCOx9uExg)?
```
# Load TFLite model and allocate tensors
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Read image and decode 
# code is different than notebook - tf.io.read_file is tf 2.3 so perhaps newer
img = tf.io.read_file(filename)
img_tensor = tf.image.decode_image(img)

# Preprocess image
img_tensor = tf.image.resize(img_tensor, size)
img_tensor = tf.cast(img_tensor, tf.uint8)
  
# Point the data to be used for testing and run the interpreter
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Obtain results and map them to the classes
predictions = interpreter.get_tensor(output_details[0]['index'])[0]

# Get indices of the top k results
_, top_k_indices = tf.math.top_k(predictions, k=top_k_results)

for i in range(top_k_results):
	print(labels[top_k_indices[i]], predictions[top_k_indices[i]] / 255.0)
```

### Object Detection 
- Detect 



# Course 3 - Data Pipelines
## 

# Course 4 - Advanced Deployment Scenarios
## 
<!--stackedit_data:
eyJoaXN0b3J5IjpbMjA4ODQ0MDY5NSwtMjA2MTUxNjQ5MCwtMT
E0MTQ3NDg1LDgxNzU5ODMwNCwtNDM4ODU2MzAyLDExMjE3OTI0
MjAsMjAyMDM3Nzk3MywtMTI2MjMwNDMzNiwtNDMyNjczMjcsMT
gxNzgyMDYyLC0xNzA4NTk4MDgsMTk1MzA1NDQ1NiwtNjQxNjE5
MTQyLDEwNDA4MTkxMywxOTgyMDgxOTEsNzQ0MTg4MTYzLDEwNj
c2NjE3NTksLTk0Mjc2MDIxOCw0NzM1MjIyNjcsLTk3NDA3NzYw
Ml19
-->