# Specialization TF Deployment

# Course 1 - Browser Based Models with TF.js

## [Week 1 - simple NN in js](https://www.coursera.org/learn/browser-based-models-tensorflow/home/week/1)

[Github Repo](https://github.com/lmoroney/dlaicourse/tree/master/TensorFlow%20Deployment)
- tensorflow.js supports layers api from keras
- lower api is core.api
- core can run on node.js too
- tensor2d is used (numpy is not available in js)
- training is asynchronous and based on callbacks because we don't want to lock the browser


### Simple Model Training

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

<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE2NzcyMzQ3NTgsMjAyODQ1Nzc5M119
-->