# Membership Inference Attack on Neural Networks

**Authors:**  
Navaneeth S (s3500721), Adithya Pradeep Varma (s3350363)

## Abstract  
This report explains the implementation and application of a Membership Inference Attack (MIA) on a neural network model using confidence score and entropy-based ranking. The goal is to determine which samples from a dataset are most likely to have been part of the training data. The attack leverages the differences in model behavior between training and non-training samples.

## Introduction  
Membership Inference Attacks (MIAs) are a form of privacy attack on machine learning models where an adversary attempts to determine if a specific data sample was used in the model's training set. This type of attack exploits the confidence scores or entropy of the model's predictions to infer membership. MIAs have significant implications for data privacy, especially when applied to sensitive datasets.

In this report, we describe the technique used to perform an MIA on a TensorFlow model trained on the Texas-100 dataset, detailing the implementation, methodology, and results.

## Methodology  
The attack was performed in the following steps:

### Step 1: Dataset Preparation  
The dataset used for this attack is the Texas-100 dataset. The first 200 samples of the dataset were loaded and preprocessed:
- Features were extracted from the `feats` file using NumPy.
- Labels were loaded from the `labels` file.
- Only the first 200 samples were used for the attack.

### Step 2: Model Loading  
The pre-trained TensorFlow model was loaded using the 'tf.saved_model' module. The inference function was extracted from the `serving_default` signature to make predictions on the prepared dataset. The output tensor was dynamically determined to ensure compatibility with the model's structure.

### Step 3: Prediction and Ranking  
The attack methodology involved the following calculations:
- **Confidence Calculation:** The maximum softmax probability was calculated for each sample to determine the confidence score.
- **Entropy Calculation:** The entropy of the softmax output was computed to measure the uncertainty of the prediction.
- **Ranking Score:** The ranking score for each sample was computed by subtracting entropy from confidence, prioritizing samples with high confidence and low uncertainty. Samples having high confidence and low entropy are highly likely to have been part of the training dataset.

The top 100 samples with the highest ranking scores were identified as the most likely members of the training set.

### Step 4: Results and Report Generation  
The indices of the top 100 samples were saved to a file (`members.txt`). Additionally, a brief report summarizing the attack methodology and results was generated (`report.txt`).

## Implementation Details  
The attack was implemented using Python with the following key libraries:
- **NumPy:** For data manipulation and numerical calculations.
- **TensorFlow:** For loading the pre-trained model and performing predictions.
- **SciPy:** For computing entropy of the softmax outputs.

The code was structured into clearly defined steps, ensuring modularity and clarity. Below is a simplified code snippet highlighting the core functionality:

```python
# Load the model
model = tf.saved_model.load(model_path)
infer = model.signatures["serving_default"]

# Make predictions
features_tensor = tf.constant(features, dtype=tf.float32)
predictions = infer(features_tensor)[output_tensor_name].numpy()

# Compute confidence and entropy
confidences = np.max(predictions, axis=1)
entropies = np.apply_along_axis(lambda x: entropy(x), 1, predictions)

# Ranking score and selection
ranking_score = confidences - entropies
member_indices = np.argsort(ranking_score)[-100:]

## Results and Analysis
The Membership Inference Attack successfully identified the top 100 samples that were most likely part of the training set. The confidence and entropy-based ranking method proved effective, with the results saved in members.txt. The generated report summarizes the methodology and findings.

## Conclusion
Membership Inference Attacks are a critical reminder of the privacy risks associated with machine learning models. This report demonstrates the effectiveness of confidence and entropy-based ranking in performing such attacks. The success of this attack emphasizes the need for increased privacy measures and techniques to mitigate the risk of such privacy breaches in machine learning models.