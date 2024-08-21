// Import TensorFlow.js
import * as tf from '@tensorflow/tfjs';

// Define a more complex model
const model = tf.sequential();

// Input layer
model.add(tf.layers.dense({inputShape: [6], units: 128, activation: 'relu'}));

// Hidden layers
model.add(tf.layers.dense({units: 64, activation: 'relu'}));
model.add(tf.layers.dense({units: 32, activation: 'relu'}));

// Optional dropout layer for regularization
model.add(tf.layers.dropout({rate: 0.2}));

// Output layer for PDI, Particle Size, Zeta Potential
model.add(tf.layers.dense({units: 3}));

// Compile the model
model.compile({
    optimizer: 'adam',
    loss: 'meanSquaredError',
    metrics: ['mse']
});

// Function to predict using the model
async function makePrediction(inputData) {
    const inputTensor = tf.tensor2d([inputData], [1, 6]);  // Convert to 2D tensor with shape [1, 6])
    const prediction = model.predict(inputTensor);
    const predictionData = await prediction.data();
    return Array.from(predictionData);  // Convert the tensor to a regular array
}

// Function to train the model (dummy training for example)
async function trainModel() {
    // Generate some random dummy data for training
    const xs = tf.randomNormal([100, 6]);  // 100 samples of 6 input parameters
    const ys = tf.randomNormal([100, 3]);  // 100 samples of 3 output parameters (PDI, Particle Size, Zeta Potential)

    await model.fit(xs, ys, {
        epochs: 100,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                console.log(`Epoch ${epoch}: loss = ${logs.loss}`);
            }
        }
    });

    console.log('Model training complete.');
}

// Ensure the model is trained before making predictions
await trainModel();

export { makePrediction };
