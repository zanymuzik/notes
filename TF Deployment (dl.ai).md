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

# Course 2 - Device based Models with TFLi
## [Javascript in TensorFlow](https://www.tensorflow.org/js)

# Course 3 - Browser Based Models with TF.js
## [Javascript in TensorFlow](https://www.tensorflow.org/js)

# Course 4 - Browser Based Models with TF.js
## [Javascript in TensorFlow](https://www.tensorflow.org/js)

<!--stackedit_data:
eyJoaXN0b3J5IjpbLTM2NDEzMjc2LDQ3MDg3MzAxNyw0NjU0MD
IxMTUsMTU2NjgxMjE2LC0xMzMzNjAxMzI2LDEzMDY4MDc0Myw2
MzI2NTY1NjQsLTY4MjU5MzkzNSwtNjYwNTYwNzY5LDIwMjg0NT
c3OTNdfQ==
-->