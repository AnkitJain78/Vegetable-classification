const express = require('express');
const multer = require('multer');
const cors = require("cors")
const fs = require("fs/promises")
const tf = require('@tensorflow/tfjs-node');

const app = express();
const port = 5000;

app.use(cors())

const upload = multer({ dest: 'uploads/' });

const modelPath = './vegetable_model/model.json';

let model;

const classMap = {0: 'Bean', 1: 'Bitter_Gourd', 2: 'Bottle_Gourd', 3: 'Brinjal', 4: 'Broccoli', 5: 'Cabbage', 6: 'Capsicum', 7: 'Carrot', 8: 'Cauliflower', 9: 'Cucumber', 10: 'Papaya', 11: 'Potato', 12: 'Pumpkin', 13: 'Radish', 14: 'Tomato'}


async function loadModel() {
  try {
    model = await tf.loadLayersModel(`file://${modelPath}`);
    console.log('Model loaded successfully.');
  } catch (error) {
    console.error('Error loading model:', error);
  }
}

loadModel();

async function predictImage(testImagePath) {
  try {
    const data = await fs.readFile(testImagePath);
    const imageTensor = tf.node.decodeImage(data);
    const resizedImage = tf.image.resizeBilinear(imageTensor, [150, 150]);
    const testImage = resizedImage.div(255);
    const testImageInput = testImage.expandDims(0);
    const prediction = await model.predict(testImageInput).data();
    const predictedLabelIndex = tf.argMax(prediction).dataSync()[0];
    return {result: classMap[predictedLabelIndex]}
  } catch (error) {
    console.error('Error generating predictions:', error);
  }
}

app.post('/upload', upload.single('image'), async (req, res) => {
    if (!req.file) {
      return res.status(400).send('No file uploaded.');
    }
    const prediction = await predictImage(req.file.path);
    res.json({ prediction });
});

app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});
