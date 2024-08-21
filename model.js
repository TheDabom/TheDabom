// Import TensorFlow.js library
import * as tf from '@tensorflow/tfjs';

// Define a simple neural network model
const model = tf.sequential();

// Add layers to the model
model.add(tf.layers.dense({inputShape: [6], units: 32, activation: 'relu'}));
model.add(tf.layers.dense({units: 16, activation: 'relu'}));
model.add(tf.layers.dense({units: 3})); // Output layer for PDI, Particle Size, and Zeta Potential

// Compile the model
model.compile({
    optimizer: 'adam',
    loss: 'meanSquaredError',
    metrics: ['mse']
});

// Function to predict using the model
async function makePrediction(inputData) {
    const inputTensor = tf.tensor2d([inputData], [1, 6]); // Reshape to 2D tensor
    const prediction = model.predict(inputTensor);
    const predictionData = await prediction.data();
    return Array.from(predictionData);
}

// Train the model with dummy data (you would replace this with actual data)
async function trainModel() {
    const xs = tf.randomNormal([100, 6]); // Replace with actual input data
    const ys = tf.randomNormal([100, 3]); // Replace with actual output data

    await model.fit(xs, ys, {
        epochs: 100,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                console.log(`Epoch ${epoch}: loss = ${logs.loss}`);
            }
        }
    });
}

// Train the model when the page loads
trainModel();

export { makePrediction };
