/**
 * @file main.js
 * @description Implements a CycleGAN model in TensorFlow.js for image-to-image translation.
 * This script includes functionality for building the generator and discriminator,
 * data loading and augmentation, the complete training loop with progressive resizing,
 * and a user interface for training, model management, and image enhancement.
 *
 * The implementation details (e.g., loss functions, learning rate schedule) are
 * aligned with the original CycleGAN paper (arXiv:1703.10593v7).
 */

// --- Training Hyperparameters ---
const EPOCHS = 200; // Total number of training epochs, as per the paper.
const ITERS = 100; // Total number of iters per epoch, as per the official pytorch implementation
const BATCH_SIZE = 1; // Number of images to process in a single batch.

// --- Progressive Training Settings ---
const START_SIZE = 32; // Starting image resolution.
const END_SIZE = 256; // Final image resolution.
const SIZE_STEP = 16; // Pixel step for each resolution increase.
const ENHANCE_SIZE = 256; // Fixed size for the enhancement feature.

// --- Optimizer Settings ---
const INITIAL_LR = 2e-4; // Initial learning rate for the Adam optimizer.
const ADAM_BETA1 = 0.5; // Exponential decay rate for the first moment estimates in Adam.
const ADAM_BETA2 = 0.999; // Exponential decay rate for the second moment estimates in Adam.

// --- Model Save Paths ---
const MODEL_PATH_GEN_A2B = 'indexeddb://cyclegan_genA2B_final';
const MODEL_PATH_GEN_B2A = 'indexeddb://cyclegan_genB2A_final';

// --- UI Element References ---
let statusElement;
let epochStatusElement;
let lossStatusElement;
let sampleContainer;
let saveModelBtn;
let loadModelInput;
let startTrainingBtn;
let pauseResumeTrainingBtn;
let deleteModelBtn;
let stopTrainingBtn;
let trainingTimeElement;
let epochTimingElement;
let etaTimeElement;
let enhanceImageInput;
let enhanceImageBtn;
let enhanceResultContainer;
let enhanceProcessingOverlay;
let enhanceEtaElement;
let closeOverlayBtn;
let spinnerElement;

// --- Chart.js Visualization Elements ---
let lossChartCanvas;
let lossChart;
let generatorLossData = [];
let discriminatorLossData = [];
let epochLabels = [];

// --- Training Control Flags ---
let stopTrainingFlag = false;
let pauseTrainingFlag = false;

// --- Backend Management ---
let currentBackend = '';

// --- Model Reference ---
let generatorModel;

/**
 * A buffer to store a history of generated images.
 * This is used to update the discriminators using a history of images
 * rather than just the latest ones, which helps stabilize training as described in the CycleGAN paper.
 */
class ImageHistoryBuffer {
    constructor(bufferSize) {
        this.bufferSize = bufferSize;
        this.numImages = 0;
        this.images = [];
    }

    /**
     * Queries the buffer with a new image.
     * With 50% probability, it returns a random image from the history and replaces it.
     * Otherwise, it returns the provided image directly.
     * @param {tf.Tensor} image The newly generated image tensor.
     * @returns {tf.Tensor} An image tensor (either from history or the input).
     */
    query(image) {
        if (this.bufferSize === 0) {
            return image;
        }
        if (this.numImages < this.bufferSize) {
            // Use tf.keep() to prevent tf.tidy from disposing the tensor being stored in the buffer.
            const imageClone = tf.keep(image.clone());
            this.images.push(imageClone);
            this.numImages++;
            return image;
        } else {
            if (Math.random() > 0.5) {
                const randomIndex = Math.floor(Math.random() * this.bufferSize);
                // Get the tensor to return before replacing it in the buffer.
                const imageToReturn = this.images[randomIndex];
                // Use tf.keep() on the new clone before storing it.
                const imageClone = tf.keep(image.clone());
                this.images[randomIndex] = imageClone;
                // Return the old tensor but mark it for disposal by caller
                return tf.keep(imageToReturn);
            } else {
                return image;
            }
        }
    }

    /**
     * Disposes of all tensors stored in the buffer to prevent memory leaks.
     */
    dispose() {
        for (const img of this.images) {
            if (img && !img.isDisposed) {
                img.dispose();
            }
        }
        this.images = [];
        this.numImages = 0;
    }
}


// --- Custom Layer: Instance Normalization ---
class InstanceNorm extends tf.layers.Layer {
    constructor() {
        super({});
    }

    call(inputs) {
        return tf.tidy(() => {
            // Instance Normalization is a key component for style transfer models.
            const moments = tf.moments(inputs[0], [1, 2], true);
            return inputs[0].sub(moments.mean).div(moments.variance.sqrt().add(1e-5));
        });
    }

    computeOutputShape(inputShape) {
        return inputShape;
    }

    static get className() {
        return 'InstanceNorm';
    }
}


// --- CycleGAN Specific Functions ---

/**
 * Defines a residual block, a core component of the generator architecture.
 * @param {tf.Tensor} input The input tensor to the block.
 * @param {number} filters The number of filters for the convolutional layers.
 * @returns {tf.Tensor} The output tensor of the residual block.
 */
function resBlock(input, filters) {
    return tf.tidy(() => {
        // The paper's generator uses residual blocks with 3x3 convolutions.
        let x = tf.layers.conv2d({
            filters: filters,
            kernelSize: 3,
            strides: 1,
            padding: 'same'
        }).apply(input);
        x = (new InstanceNorm()).apply(x);
        x = tf.layers.reLU().apply(x); // Use ReLU activation in the generator.
        x = tf.layers.conv2d({
            filters: filters,
            kernelSize: 3,
            strides: 1,
            padding: 'same'
        }).apply(x);
        x = (new InstanceNorm()).apply(x);

        return tf.layers.add().apply([input, x]);
    });
}

/**
 * Builds a U-Net based generator for the CycleGAN that accepts dynamic input sizes.
 * @returns {tf.Model} A TensorFlow.js model representing the generator.
 */
function buildCycleGANGenerator() {
    // Define an input that can accept images of any height and width.
    const inputs = tf.input({
        shape: [null, null, 3]
    });

    // The rest of the function remains identical...
    const padding = 'same';

    // Encoder part of the generator
    let x = tf.layers.conv2d({
        filters: 64,
        kernelSize: 7,
        strides: 1,
        padding: padding
    }).apply(inputs);
    x = (new InstanceNorm()).apply(x);
    x = tf.layers.reLU().apply(x);

    // Downsampling layers
    x = tf.layers.conv2d({
        filters: 128,
        kernelSize: 3,
        strides: 2,
        padding: padding
    }).apply(x);
    x = (new InstanceNorm()).apply(x);
    x = tf.layers.reLU().apply(x);

    x = tf.layers.conv2d({
        filters: 256,
        kernelSize: 3,
        strides: 2,
        padding: padding
    }).apply(x);
    x = (new InstanceNorm()).apply(x);
    x = tf.layers.reLU().apply(x);

    // Transformation part with residual blocks
    for (let i = 0; i < 9; i++) {
        x = resBlock(x, 256);
    }

    // Decoder part with upsampling layers
    x = tf.layers.conv2dTranspose({
        filters: 128,
        kernelSize: 3,
        strides: 2,
        padding: padding
    }).apply(x);
    x = (new InstanceNorm()).apply(x);
    x = tf.layers.reLU().apply(x);

    x = tf.layers.conv2dTranspose({
        filters: 64,
        kernelSize: 3,
        strides: 2,
        padding: padding
    }).apply(x);
    x = (new InstanceNorm()).apply(x);
    x = tf.layers.reLU().apply(x);

    // Output layer
    x = tf.layers.conv2d({
        filters: 3,
        kernelSize: 7,
        strides: 1,
        padding: padding,
        activation: 'tanh'
    }).apply(x);

    return tf.model({
        inputs: inputs,
        outputs: x,
        name: 'CycleGAN_Generator'
    });
}


/**
 * Builds a CycleGAN discriminator that accepts dynamic input sizes.
 * @returns {tf.Model} A TensorFlow.js model representing the discriminator.
 */
function buildCycleGANDiscriminator() {
    // Define an input that can accept images of any height and width.
    const inputs = tf.input({
        shape: [null, null, 3]
    });
    
    // The rest of the function remains identical...
    const padding = 'same';
    let x = tf.layers.conv2d({
        filters: 64,
        kernelSize: 4,
        strides: 2,
        padding: padding
    }).apply(inputs);
    x = tf.layers.leakyReLU({ alpha: 0.2 }).apply(x);

    x = tf.layers.conv2d({
        filters: 128,
        kernelSize: 4,
        strides: 2,
        padding: padding
    }).apply(x);
    x = (new InstanceNorm()).apply(x);
    x = tf.layers.leakyReLU({ alpha: 0.2 }).apply(x);

    x = tf.layers.conv2d({
        filters: 256,
        kernelSize: 4,
        strides: 2,
        padding: padding
    }).apply(x);
    x = (new InstanceNorm()).apply(x);
    x = tf.layers.leakyReLU({ alpha: 0.2 }).apply(x);

    x = tf.layers.conv2d({
        filters: 512,
        kernelSize: 4,
        strides: 1,
        padding: padding
    }).apply(x);
    x = (new InstanceNorm()).apply(x);
    x = tf.layers.leakyReLU({ alpha: 0.2 }).apply(x);

    const output = tf.layers.conv2d({
        filters: 1,
        kernelSize: 4,
        strides: 1,
        padding: padding
    }).apply(x);

    return tf.model({
        inputs: inputs,
        outputs: output,
        name: 'CycleGAN_Discriminator'
    });
}


/**
 * Applies minimal data ntation: a random horizontal flip.
 * The original paper does not specify extensive augmentation, so this is kept simple.
 * @param {tf} tf - The TensorFlow.js library object.
 * @param {tf.Tensor} tensor - The input image tensor [H, W, C].
 * @returns {tf.Tensor} The augmented image tensor.
 */
function cycleGANAugment(tf, tensor) {
    return tf.tidy(() => {
        let img = tensor;

        if (img.rank === 3) {
            img = img.expandDims(0);
        }

        // Randomly flip the image horizontally.
        if (Math.random() > 0.5) {
            img = tf.image.flipLeftRight(img);
        }

        return img.squeeze();
    });
}


/**
 * Loads and preprocesses a single image for the CycleGAN model.
 * It resizes the image to the target size and normalizes pixel values to [-1, 1].
 * @param {tf} tf - The TensorFlow.js library object.
 * @param {string} imagePath - The URL or path to the image.
 * @param {number} targetSize - The desired output size. Defaults to 256.
 * @returns {Promise<tf.Tensor>} A promise that resolves to the preprocessed image tensor.
 */
async function loadCycleGANImage(tf, imagePath, targetSize = 256) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = "anonymous"; // Enable loading images from other domains.

        img.onload = () => {
            try {
                const tensor = tf.tidy(() => {
                    const pixels = tf.browser.fromPixels(img);
                    // Normalize pixel values from [0, 255] to [-1, 1].
                    const normalized = pixels.toFloat().div(127.5).sub(1);
                    const resized = tf.image.resizeBilinear(normalized, [targetSize, targetSize]);
                    return resized;
                });
                resolve(tensor);
            } catch (err) {
                console.error(`[TensorFlow Error] Could not process image: ${imagePath} ❌`, err);
                reject(err);
            }
        };

        img.onerror = (e) => {
            console.error(`[Image Load Error] Failed to load image from path: ${imagePath} ❌`, e);
            reject(new Error(`Image load failed: ${imagePath}`));
        };

        img.src = imagePath;
    });
}



/**
 * Loads a batch of augmented images from a specified domain (e.g., 'A' for horses).
 * @param {tf} tf - The TensorFlow.js library object.
 * @param {string} domain - The domain to load from ('A' or 'B').
 * @param {number} batchSize - The number of images in the batch.
 * @param {number} imageSize - The target size for the images.
 * @returns {Promise<tf.Tensor>} A promise that resolves to a batch tensor of shape [batchSize, H, W, C].
 */
async function loadDomainBatch(tf, domain, batchSize, imageSize) {
    const images = [];
    const usedIndices = new Set(); // Ensure no duplicate images are loaded in a single batch.

    for (let i = 0; i < batchSize; i++) {
        let index;
        // Select a random image index that hasn't been used yet.
        do {
            index = Math.floor(Math.random() * 500) + 1; // Assuming 500 images in the dataset.
        } while (usedIndices.has(index));
        usedIndices.add(index);

        const paddedIndex = index.toString().padStart(3, '0');
        const imagePath = `./horse2zebra/train${domain}/${paddedIndex}.jpeg`;

        try {
            const tensor = await loadCycleGANImage(tf, imagePath, imageSize);
            const augmented = cycleGANAugment(tf, tensor); // Use the simplified augmentation
            images.push(augmented);
            tensor.dispose();
        } catch (err) {
            console.warn(`Skipping failed image load: ${imagePath}`, err);
            i--; // Retry loading a different sample.
        }
    }

    if (images.length === 0) {
        throw new Error("Failed to load any images for the batch. Check image paths and network.");
    }

    // Stack individual image tensors into a single batch tensor.
    const batchTensor = tf.stack(images);
    images.forEach(t => t.dispose()); // Clean up the array of tensors.
    return batchTensor;
}


// --- CycleGAN Loss Functions ---

/**
 * Calculates the generator's loss using the Least Squares GAN (LSGAN) objective.
 * The goal is to make the discriminator output 1 for fake images.
 * @param {tf.Tensor} fake - The discriminator's output for fake (generated) images.
 * @returns {tf.Tensor} The generator's adversarial loss.
 */
function lsganLossGenerator(fake) {
    return tf.tidy(() =>
        tf.mean(tf.squaredDifference(fake, tf.ones(fake.shape)))
    );
}

/**
 * Calculates the discriminator's loss using the LSGAN objective.
 * The goal is to make the discriminator output 1 for real images and 0 for fake images.
 * @param {tf.Tensor} real - The discriminator's output for real images.
 * @param {tf.Tensor} fake - The discriminator's output for fake images.
 * @returns {tf.Tensor} The discriminator's adversarial loss.
 */
function lsganLossDiscriminator(real, fake) {
    return tf.tidy(() => {
        const realLoss = tf.mean(tf.squaredDifference(real, tf.ones(real.shape)));
        const fakeLoss = tf.mean(tf.squaredDifference(fake, tf.zeros(fake.shape)));
        // The paper divides the discriminator objective by 2 to slow its learning rate relative to the generator.
        return realLoss.add(fakeLoss).mul(0.5);
    });
}


/**
 * Calculates the cycle-consistency loss.
 * This L1 loss ensures that if an image is translated and then translated back,
 * it should resemble the original image.
 * @param {tf.Tensor} real - The original image tensor.
 * @param {tf.Tensor} reconstructed - The cycle-reconstructed image tensor.
 * @param {number} lambda - The weight of the cycle loss. Defaults to 10.
 * @returns {tf.Tensor} The weighted cycle-consistency loss.
 */
function cycleLoss(real, reconstructed, lambda = 10) {
    return tf.mul(lambda, tf.losses.absoluteDifference(real, reconstructed));
}

/**
 * Calculates the identity loss.
 * This loss ensures that when the generator is fed an image from the target domain,
 * it should output something close to the input (identity mapping).
 * @param {tf.Tensor} real - The real image tensor.
 * @param {tf.Tensor} identity - The identity-mapped image tensor.
 * @param {number} lambda - The weight of the identity loss. Defaults to 5 (0.5 * cycle_lambda).
 * @returns {tf.Tensor} The weighted identity loss.
 */
function identityLoss(real, identity, lambda = 5) {
    return tf.mul(lambda, tf.losses.absoluteDifference(real, identity));
}


// --- Utility Functions ---

/**
 * Updates the main status message in the UI.
 * @param {string} message - The message to display.
 */
function updateStatus(message) {
    if (statusElement) {
        statusElement.textContent = `Status: ${message}`;
    }
}

/**
 * Generates a timestamp string for logging purposes.
 * @returns {string} The current time in HH:MM:SS format.
 */
function getTimestamp() {
    return new Date().toTimeString().split(' ')[0];
}

/**
 * Logs the current TensorFlow.js memory usage to the console.
 * @param {tf} tf - The TensorFlow.js library object.
 */
function logMemoryUsage(tf) {
    if (tf && tf.getBackend() && tf.memory) {
        const mem = tf.memory();
        console.log(`[${getTimestamp()}] TF Memory: ${(mem.numBytes / 1024 / 1024).toFixed(2)} MB (${mem.numTensors} tensors)`);
    }
}

/**
 * Displays a tensor as an image by drawing it to a canvas and appending it to a parent element.
 * @param {tf} tf - The TensorFlow.js library object.
 * @param {tf.Tensor} tensor - The image tensor to display, normalized to [-1, 1].
 * @param {HTMLElement} parentElement - The DOM element to append the canvas to.
 * @param {string} title - A title to display above the image.
 */
async function displayTensorAsImage(tf, tensor, parentElement, title) {
    // Tidy to manage memory of intermediate tensors
    const displayTensor = tf.tidy(() =>
        tensor.squeeze().add(1).div(2).clipByValue(0, 1) // Renormalize from [-1, 1] to [0, 1]
    );
    const canvas = document.createElement('canvas');
    canvas.width = displayTensor.shape[1];
    canvas.height = displayTensor.shape[0];
    await tf.browser.toPixels(displayTensor, canvas);

    const container = document.createElement('div');
    container.style.display = 'inline-block';
    container.style.margin = '10px';
    container.style.flexShrink = '0'; // Prevent images from shrinking in the flex container
    const h4 = document.createElement('h4');
    h4.textContent = title;
    container.appendChild(h4);
    container.appendChild(canvas);
    parentElement.appendChild(container);

    displayTensor.dispose(); // Dispose of the tensor after drawing
}

/**
 * Initializes the most performant available TensorFlow.js backend (WebGPU > WebGL).
 */
async function initializeTfBackend() {
/*
    updateStatus('Initializing backend: trying WebGPU...');
    try {
        await tf.setBackend('webgpu');
        currentBackend = 'webgpu';
        updateStatus('Backend initialized: WebGPU.');
        console.log(`Successfully set backend to: ${tf.getBackend()}`);
        return;
    } catch (error) {
        console.warn('WebGPU initialization failed. Falling back to WebGL.', error);
    }
*/
    updateStatus('Initializing backend: trying WebGL...');
    try {
        await tf.setBackend('webgl');
        currentBackend = 'webgl';
        updateStatus('Backend initialized: WebGL.');
        console.log(`Successfully set backend to: ${tf.getBackend()}`);
        return;
    } catch (error) {
        console.error('WebGL initialization also failed. No suitable GPU backend found.', error);
        updateStatus('Error: No GPU backend (WebGPU or WebGL) is available. Performance will be limited.');
    }
}


/**
 * Initializes the Chart.js instance for displaying the loss curves.
 */
function initializeLossChart() {
    if (lossChart) lossChart.destroy(); // Clear previous chart if it exists
    const ctx = lossChartCanvas.getContext('2d');
    lossChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: epochLabels,
            datasets: [{
                label: 'Generator Loss',
                data: generatorLossData,
                borderColor: 'rgb(75, 192, 192)',
                tension: 0.1,
                fill: false
            }, {
                label: 'Discriminator Loss',
                data: discriminatorLossData,
                borderColor: 'rgb(255, 99, 132)',
                tension: 0.1,
                fill: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Epoch'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Loss'
                    },
                    beginAtZero: true
                }
            }
        }
    });
}

/**
 * Appends new data points to the loss chart and updates it.
 * @param {number} iteration - The current epoch or iteration number.
 * @param {number} genLoss - The generator loss value for this iteration.
 * @param {number} discLoss - The discriminator loss value for this iteration.
 */
function updateLossChart(iteration, genLoss, discLoss) {
    epochLabels.push(`Epoch ${iteration}`);
    generatorLossData.push(genLoss);
    discriminatorLossData.push(discLoss);
    lossChart.update();
}

/**
 * Clears all data from the loss chart and re-initializes it.
 */
function resetLossChart() {
    generatorLossData = [];
    discriminatorLossData = [];
    epochLabels = [];
    if (lossChart) {
        lossChart.destroy();
    }
    initializeLossChart();
}

/**
 * Calculates the input size for a given epoch during progressive training.
 * @param {number} currentEpoch - The current training epoch (0-indexed).
 * @param {number} maxEpoch - The total number of epochs.
 * @returns {number} The calculated square image size for the current epoch.
 */
function calculateInputSize(currentEpoch, maxEpoch) {
    const numSteps = (END_SIZE - START_SIZE) / SIZE_STEP;
    if (numSteps <= 0) return END_SIZE;

    // Ensure there's at least one epoch per step to avoid division by zero
    const epochsPerStep = Math.max(1, Math.floor(maxEpoch / numSteps));

    // Calculate how many sizing steps should have been taken by the current epoch
    const stepsTaken = Math.floor(currentEpoch / epochsPerStep);
    const currentSize = START_SIZE + (stepsTaken * SIZE_STEP);

    // Clamp the size to the maximum defined end size
    return Math.min(currentSize, END_SIZE);
}


/**
 * Training loop with updated iteration rate (-1 every 2 epochs) and layout fixes.
 * This version initializes dynamic models once and reuses them to avoid recompilation.
 */
async function runTraining() {
    resetLossChart();
    stopTrainingFlag = false;
    pauseTrainingFlag = false;

    // --- UI Setup for Training ---
    startTrainingBtn.disabled = true;
    pauseResumeTrainingBtn.style.display = 'inline-block';
    pauseResumeTrainingBtn.textContent = 'Pause Training';
    pauseResumeTrainingBtn.disabled = false;
    stopTrainingBtn.disabled = false;
    updateStatus("Initializing models for training...");

    // --- Model and Optimizer Initialization (Done ONCE) ---
    // Models are now built once with dynamic inputs and are reused.
    const genA2B = buildCycleGANGenerator();
    const genB2A = buildCycleGANGenerator();
    const discA = buildCycleGANDiscriminator();
    const discB = buildCycleGANDiscriminator();

    const genOptimizer = tf.train.adam(INITIAL_LR, ADAM_BETA1, ADAM_BETA2);
    const discOptimizer = tf.train.adam(INITIAL_LR, ADAM_BETA1, ADAM_BETA2);
    
    let currentSize = -1; // Still used for data loading and logging

    const fakeABuffer = new ImageHistoryBuffer(50);
    const fakeBBuffer = new ImageHistoryBuffer(50);
    const startTime = performance.now();

    // --- ETA Calculation Variable ---
    let totalWorkUnitTime = 0;

    // --- Main Training Loop ---
    for (let epoch = 0; epoch < EPOCHS; epoch++) {
        if (stopTrainingFlag) {
            updateStatus(`Training stopped by user before epoch ${epoch + 1}.`);
            break;
        }

        while (pauseTrainingFlag) {
            updateStatus("Training Paused. Click 'Resume Training' to continue.");
            await new Promise(resolve => setTimeout(resolve, 500));
        }

        const newSize = calculateInputSize(epoch, EPOCHS);
        if (newSize !== currentSize) {
            currentSize = newSize;
            updateStatus(`Epoch ${epoch + 1}: Training with new size ${currentSize}x${currentSize}...`);
        }
        
        // Iterations now decrease by 1 every 2 epochs
        const currentIters = Math.max(1, ITERS - Math.floor(epoch / 2));

        // --- THE MODEL REBUILDING BLOCK IS REMOVED ---
        // The same models (genA2B, discA, etc.) are used for all epochs and sizes.

        // --- Learning Rate Decay ---
        const DECAY_START_EPOCH = 100;
        if (epoch >= DECAY_START_EPOCH) {
            const decayFactor = 1.0 - (epoch - DECAY_START_EPOCH) / (EPOCHS - DECAY_START_EPOCH);
            const newLr = INITIAL_LR * decayFactor;
            genOptimizer.learningRate = newLr;
            discOptimizer.learningRate = newLr;
        }

        const genVars = [...genA2B.trainableWeights, ...genB2A.trainableWeights].map(v => v.val);
        const discVars = [...discA.trainableWeights, ...discB.trainableWeights].map(v => v.val);

        const epochStart = performance.now();
        let epochGenLoss = 0;
        let epochDiscLoss = 0;
        let realA = null,
            realB = null;

        // --- Iterations per Epoch Loop (uses dynamic currentIters) ---
        for (let iter = 0; iter < currentIters; iter++) {
            if (stopTrainingFlag) break;
            while (pauseTrainingFlag) {
                updateStatus("Training Paused. Click 'Resume Training' to continue.");
                await new Promise(resolve => setTimeout(resolve, 500));
            }

            epochStatusElement.textContent = `Epoch: ${epoch + 1}/${EPOCHS} (Size: ${currentSize}px) | Iter: ${iter + 1}/${currentIters}`;

            try {
                const oldRealA = realA;
                const oldRealB = realB;
                [realA, realB] = await Promise.all([
                    loadDomainBatch(tf, 'A', BATCH_SIZE, currentSize),
                    loadDomainBatch(tf, 'B', BATCH_SIZE, currentSize)
                ]);
                if (oldRealA) oldRealA.dispose();
                if (oldRealB) oldRealB.dispose();

                const genLossValue = genOptimizer.minimize(() => {
                    return tf.tidy(() => {
                        const fakeB = genA2B.apply(realA);
                        const fakeA = genB2A.apply(realB);
                        return lsganLossGenerator(discB.apply(fakeB))
                            .add(lsganLossGenerator(discA.apply(fakeA)))
                            .add(cycleLoss(realA, genB2A.apply(fakeB)))
                            .add(cycleLoss(realB, genA2B.apply(fakeA)))
                            .add(identityLoss(realA, genB2A.apply(realA)))
                            .add(identityLoss(realB, genA2B.apply(realB)));
                    });
                }, true, genVars);

                const discLossValue = discOptimizer.minimize(() => {
                    return tf.tidy(() => {
                        const fakeB_raw = genA2B.apply(realA);
                        const fakeA_raw = genB2A.apply(realB);
                        const fakeB = fakeBBuffer.query(fakeB_raw);
                        const fakeA = fakeABuffer.query(fakeA_raw);
                        const discLossA = lsganLossDiscriminator(discA.apply(realA), discA.apply(fakeA));
                        const discLossB = lsganLossDiscriminator(discB.apply(realB), discB.apply(fakeB));
                        if (fakeA !== fakeA_raw) fakeA.dispose();
                        if (fakeB !== fakeB_raw) fakeB.dispose();
                        return discLossA.add(discLossB);
                    });
                }, true, discVars);

                const [iterGenLoss, iterDiscLoss] = await Promise.all([genLossValue.data(), discLossValue.data()]);
                epochGenLoss += iterGenLoss[0];
                epochDiscLoss += iterDiscLoss[0];

                genLossValue.dispose();
                discLossValue.dispose();
            } catch (error) {
                console.error(`Error during epoch ${epoch + 1}, iter ${iter + 1}:`, error);
                updateStatus(`Error in training loop: ${error.message}. Stopping.`);
                stopTrainingFlag = true;
                break;
            }
        } // --- End of Iterations Loop ---

        if (stopTrainingFlag) break;

        const epochTime = (performance.now() - epochStart) / 1000;
        const avgGenLoss = epochGenLoss / currentIters;
        const avgDiscLoss = epochDiscLoss / currentIters;

        lossStatusElement.textContent = `Avg Gen Loss: ${avgGenLoss.toFixed(4)}, Avg Disc Loss: ${avgDiscLoss.toFixed(4)}`;
        epochTimingElement.textContent = `${epochTime.toFixed(2)}s`;
        updateLossChart(epoch + 1, avgGenLoss, avgDiscLoss);

        // --- ETA logic updated for new iteration schedule ---
        const workUnitTime = epochTime / (currentIters * currentSize * currentSize);
        totalWorkUnitTime += workUnitTime;
        const avgWorkUnitTime = totalWorkUnitTime / (epoch + 1);

        let estimatedRemainingTime = 0;
        for (let futureEpoch = epoch + 1; futureEpoch < EPOCHS; futureEpoch++) {
            const futureSize = calculateInputSize(futureEpoch, EPOCHS);
            const futureIters = Math.max(1, ITERS - Math.floor(futureEpoch / 2));
            const estimatedEpochWork = futureIters * futureSize * futureSize;
            const estimatedEpochTime = avgWorkUnitTime * estimatedEpochWork;
            estimatedRemainingTime += estimatedEpochTime;
        }
        etaTimeElement.textContent = new Date(estimatedRemainingTime * 1000).toISOString().substr(11, 8);


        // --- Visualize image samples ---
        updateStatus(`Visualizing samples for epoch ${epoch + 1}...`);
        const visualizationTensors = tf.tidy(() => {
            const sampleA = realA.slice([0, 0, 0, 0], [1, -1, -1, -1]);
            const sampleB = realB.slice([0, 0, 0, 0], [1, -1, -1, -1]);
            const fakeB = genA2B.predict(sampleA);
            const fakeA = genB2A.predict(sampleB);
            const reconstructedA = genB2A.predict(fakeB);
            const reconstructedB = genA2B.predict(fakeA);
            return { sampleA, sampleB, fakeB, fakeA, reconstructedA, reconstructedB };
        });

        try {
            sampleContainer.innerHTML = '';
            // CSS Fix: Use Flexbox for a robust single-row layout
            sampleContainer.style.display = 'flex';
            sampleContainer.style.flexWrap = 'nowrap';
            sampleContainer.style.overflowX = 'auto';

            await displayTensorAsImage(tf, visualizationTensors.sampleA, sampleContainer, `Epoch ${epoch + 1} - Real A`);
            await displayTensorAsImage(tf, visualizationTensors.fakeB, sampleContainer, `Epoch ${epoch + 1} - Fake B`);
            await displayTensorAsImage(tf, visualizationTensors.reconstructedA, sampleContainer, `Epoch ${epoch + 1} - Reconstructed A`);
            await displayTensorAsImage(tf, visualizationTensors.sampleB, sampleContainer, `Epoch ${epoch + 1} - Real B`);
            await displayTensorAsImage(tf, visualizationTensors.fakeA, sampleContainer, `Epoch ${epoch + 1} - Fake A`);
            await displayTensorAsImage(tf, visualizationTensors.reconstructedB, sampleContainer, `Epoch ${epoch + 1} - Reconstructed B`);
        } catch (error) {
            console.error('Sample visualization failed:', error);
        } finally {
            Object.values(visualizationTensors).forEach(tensor => tensor.dispose());
        }

        if (realA) realA.dispose();
        if (realB) realB.dispose();

        if (epoch > 0 && epoch % 20 === 0) {
            updateStatus(`Saving models after epoch ${epoch + 1}...`);
            await genA2B.save(`indexeddb://cyclegan_genA2B_epoch_${epoch}`);
            await genB2A.save(`indexeddb://cyclegan_genB2A_epoch_${epoch}`);
            updateStatus(`Models saved after epoch ${epoch + 1}.`);
        }

        updateStatus(`Epoch ${epoch + 1}/${EPOCHS} completed.`);
        if (epoch === 0) logMemoryUsage(tf);
    } // --- End of Epochs Loop ---

    const totalTrainingTime = ((performance.now() - startTime) / 1000);
    const timeFormatted = new Date(totalTrainingTime * 1000).toISOString().substr(11, 8);
    updateStatus(`Training finished! Total time: ${timeFormatted}`);
    if (!stopTrainingFlag) {
        await genA2B.save(MODEL_PATH_GEN_A2B);
        await genB2A.save(MODEL_PATH_GEN_B2A);
    }

    startTrainingBtn.disabled = false;
    pauseResumeTrainingBtn.style.display = 'none';
    stopTrainingBtn.disabled = true;
    enhanceImageBtn.disabled = false;

    [genA2B, genB2A, discA, discB].forEach(model => model && model.dispose());
    fakeABuffer.dispose();
    fakeBBuffer.dispose();
    logMemoryUsage(tf);
}

/**
 * Saves the currently loaded generator model to a downloadable file.
 */
async function saveModelToFile() {
    let modelToSave = generatorModel;
    if (!modelToSave) {
        try {
            modelToSave = await tf.loadLayersModel(MODEL_PATH_GEN_A2B);
        } catch (e) {
            updateStatus('Cannot save. No model is currently loaded or trained.');
            return;
        }
    }

    updateStatus('Saving generator model to file...');
    try {
        await modelToSave.save('downloads://cyclegan-generator-model');
        updateStatus('Generator model has been downloaded.');
    } catch (error) {
        updateStatus(`Error saving model: ${error.message}`);
    } finally {
        if (!generatorModel) modelToSave.dispose();
    }
}

/**
 * Loads a generator model from user-selected .json and .bin files.
 * @param {Event} event - The file input change event.
 */
async function loadModelFromFile(event) {
    if (event.target.files.length === 0) return;
    updateStatus('Loading generator model from files...');

    if (generatorModel) {
        generatorModel.dispose();
        generatorModel = null;
    }

    try {
        const files = Array.from(event.target.files);
        const jsonFile = files.find(f => f.name.endsWith('.json'));
        const weightFiles = files.filter(f => f.name.endsWith('.bin'));

        if (!jsonFile || weightFiles.length === 0) {
            throw new Error("Please select both the .json and all corresponding .bin weight files.");
        }

        generatorModel = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, ...weightFiles]));
        updateStatus('Saving loaded model to browser storage for future sessions...');
        await generatorModel.save(MODEL_PATH_GEN_A2B);

        updateStatus('Generator model loaded successfully. Ready for enhancement.');
        enhanceImageBtn.disabled = false;
        generatorModel.summary();
    } catch (error) {
        updateStatus(`Error loading model from files: ${error.message}`);
        console.error('Model load error:', error);
        enhanceImageBtn.disabled = true;
    }
}

/**
 * Deletes all CycleGAN models from the browser's IndexedDB storage.
 */
async function deleteModel() {
    updateStatus('Deleting all models from browser storage...');
    try {
        const models = await tf.io.listModels();
        for (const key in models) {
            if (key.startsWith('indexeddb://cyclegan')) {
                await tf.io.removeModel(key);
                console.log(`Deleted model: ${key}`);
            }
        }

        if (generatorModel) {
            generatorModel.dispose();
            generatorModel = null;
        }

        updateStatus('All models deleted. Ready to train a new model.');
        epochStatusElement.textContent = 'Epoch: N/A';
        lossStatusElement.textContent = 'Loss: N/A';
        epochTimingElement.textContent = 'N/A';
        etaTimeElement.textContent = 'N/A';
        enhanceImageBtn.disabled = true;
    } catch (error) {
        updateStatus(`Error deleting models: ${error.message}`);
        console.error('Error during model deletion:', error);
    }
}

/**
 * Handles the image enhancement process: loads the image, runs the generator model, and displays the result.
 * @param {Event} event - The file input change event containing the image to enhance.
 */
async function enhanceImage(event) {
    const file = event.target.files[0];
    if (!file) return;

    enhanceResultContainer.innerHTML = '';
    enhanceProcessingOverlay.style.display = 'flex';
    if (spinnerElement) spinnerElement.style.display = 'block';

    if (!tf.getBackend()) {
        await initializeTfBackend();
    }

    if (generatorModel) {
        generatorModel.dispose();
        generatorModel = null;
    }

    try {
        updateStatus('Loading trained generator model...');
        generatorModel = await tf.loadLayersModel(MODEL_PATH_GEN_A2B);
        updateStatus('Model loaded. Preparing image...');

    } catch (e) {
        updateStatus('Error: Could not load a pre-trained model. Please train one first.');
        console.error("Model loading error for enhancement:", e);
        enhanceProcessingOverlay.style.display = 'none';
        if (generatorModel) generatorModel.dispose();
        return;
    }

    const startTime = performance.now();
    try {
        const uploadedImageTensor = await new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => {
                const canvas = document.createElement('canvas');
                canvas.width = ENHANCE_SIZE;
                canvas.height = ENHANCE_SIZE;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0, ENHANCE_SIZE, ENHANCE_SIZE);
                resolve(tf.browser.fromPixels(canvas).toFloat().div(127.5).sub(1));
            };
            img.onerror = reject;
            img.src = URL.createObjectURL(file);
        });

        const enhancedTensor = tf.tidy(() => {
            const batchedInput = uploadedImageTensor.expandDims(0);
            const prediction = generatorModel.predict(batchedInput);
            return prediction.squeeze();
        });

        enhanceResultContainer.innerHTML = '<h3>Enhancement Result:</h3>';
        await displayTensorAsImage(tf, uploadedImageTensor, enhanceResultContainer, `Original Input (Resized to ${ENHANCE_SIZE}x${ENHANCE_SIZE})`);
        await displayTensorAsImage(tf, enhancedTensor, enhanceResultContainer, `Generated Output (${ENHANCE_SIZE}x${ENHANCE_SIZE})`);

        enhanceEtaElement.textContent = `${((performance.now() - startTime) / 1000).toFixed(2)} seconds`;
        if (spinnerElement) spinnerElement.style.display = 'none';
        if (closeOverlayBtn) closeOverlayBtn.style.display = 'block';
        updateStatus('Image enhancement complete!');

        uploadedImageTensor.dispose();
        enhancedTensor.dispose();

    } catch (error) {
        console.error('Error during image enhancement:', error);
        updateStatus(`Error enhancing image: ${error.message}`);
        enhanceEtaElement.textContent = 'Error';
        if (spinnerElement) spinnerElement.style.display = 'none';
        if (closeOverlayBtn) closeOverlayBtn.style.display = 'block';
    } finally {
        if (generatorModel) {
            generatorModel.dispose();
            generatorModel = null;
        }
        logMemoryUsage(tf);
    }
}

// Register custom layers for serialization before any model loading or saving.
tf.serialization.registerClass(InstanceNorm);


/**
 * Main function to initialize the application after the DOM is loaded.
 */
document.addEventListener('DOMContentLoaded', async () => {
    // --- Get DOM Element References ---
    statusElement = document.getElementById('status');
    epochStatusElement = document.getElementById('epoch-status');
    lossStatusElement = document.getElementById('loss-status');
    sampleContainer = document.getElementById('sample-images').querySelector('.sample-grid');
    saveModelBtn = document.getElementById('save-model-btn');
    loadModelInput = document.getElementById('load-model-input');
    startTrainingBtn = document.getElementById('start-training-btn');
    pauseResumeTrainingBtn = document.getElementById('pause-resume-training-btn');
    deleteModelBtn = document.getElementById('delete-model-btn');
    stopTrainingBtn = document.getElementById('stop-training-btn');
    epochTimingElement = document.getElementById('epoch-time');
    etaTimeElement = document.getElementById('eta-time');
    lossChartCanvas = document.getElementById('lossChart');
    enhanceImageInput = document.getElementById('enhance-image-input');
    enhanceImageBtn = document.getElementById('enhance-image-btn');
    enhanceResultContainer = document.getElementById('enhance-results');
    enhanceProcessingOverlay = document.getElementById('enhance-processing-overlay');
    enhanceEtaElement = document.getElementById('enhance-eta');
    closeOverlayBtn = document.getElementById('close-overlay-btn');
    spinnerElement = enhanceProcessingOverlay.querySelector('.spinner');

    // --- Application Initialization ---
    initializeLossChart();
    await initializeTfBackend();

    // --- Event Listener Setup ---
    startTrainingBtn.addEventListener('click', runTraining);

    pauseResumeTrainingBtn.addEventListener('click', () => {
        pauseTrainingFlag = !pauseTrainingFlag;
        if (pauseTrainingFlag) {
            pauseResumeTrainingBtn.textContent = 'Resume Training';
            updateStatus('Training is paused.');
            stopTrainingBtn.disabled = true;
        } else {
            pauseResumeTrainingBtn.textContent = 'Pause Training';
            updateStatus('Resuming training...');
            stopTrainingBtn.disabled = false;
        }
    });

    saveModelBtn.addEventListener('click', saveModelToFile);
    loadModelInput.addEventListener('change', loadModelFromFile);
    deleteModelBtn.addEventListener('click', deleteModel);
    stopTrainingBtn.addEventListener('click', () => {
        stopTrainingFlag = true;
        updateStatus('Stopping training after the current epoch...');
        stopTrainingBtn.disabled = true;
    });

    stopTrainingBtn.disabled = true;
    pauseResumeTrainingBtn.style.display = 'none';

    try {
        const modelList = await tf.io.listModels();
        if (modelList[MODEL_PATH_GEN_A2B]) {
            enhanceImageBtn.disabled = false;
            updateStatus('Ready. Found a trained model in browser storage.');
        } else {
            enhanceImageBtn.disabled = true;
            updateStatus('Ready. No trained models found. Please train a model first.');
        }
    } catch (e) {
        console.warn("Could not check for existing models; keeping enhancement disabled.", e);
        enhanceImageBtn.disabled = true;
    }

    enhanceImageBtn.addEventListener('click', () => enhanceImageInput.click());
    enhanceImageInput.addEventListener('change', enhanceImage);

    if (enhanceProcessingOverlay) enhanceProcessingOverlay.style.display = 'none';
    if (spinnerElement) spinnerElement.style.display = 'none';
    if (closeOverlayBtn) {
        closeOverlayBtn.addEventListener('click', () => {
            enhanceProcessingOverlay.style.display = 'none';
            enhanceResultContainer.innerHTML = '';
            closeOverlayBtn.style.display = 'none';
        });
    }

    logMemoryUsage(tf);
});