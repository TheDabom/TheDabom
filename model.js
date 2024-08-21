import * as tf from '@tensorflow/tfjs';

const model = tf.sequential();

model.add(tf.layers.dense({inputShape: [6], units: 128, activation: 'relu'}));
model.add(tf.layers.dense({units: 64, activation: 'relu'}));
model.add(tf.layers.dense({units: 32, activation: 'relu'}));
model.add(tf.layers.dropout({rate: 0.2}));
model.add(tf.layers.dense({units: 3}));

model.compile({
    optimizer: 'adam',
    loss: 'meanSquaredError',
    metrics: ['mse']
});

async function makePrediction(inputData) {
    console.log("Input Data for Prediction: ", inputData);
    const inputTensor = tf.tensor2d([inputData], [1, 6]);
    console.log("Input Tensor: ", inputTensor);
    const prediction = model.predict(inputTensor);
    console.log("Raw Prediction Tensor: ", prediction);
    const predictionData = await prediction.data();
    console.log("Prediction Data: ", predictionData);
    return Array.from(predictionData);
}

async function trainModel() {
    const xs = tf.randomNormal([100, 6]);
    const ys = tf.randomNormal([100, 3]);

    await model.fit(xs, ys, {
        epochs: 10,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                console.log(`Epoch ${epoch}: loss = ${logs.loss}`);
            }
        }
    });

    console.log('Model training complete.');
}

await trainModel();

export { makePrediction };
