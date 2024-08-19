# CodeAlpha_Emotion_Recognition_from_Speech

This project involves recognizing emotions from speech audio files using a Convolutional Neural Network (CNN). The code includes data loading, feature extraction, data augmentation, model building, training, and evaluation. 

## Table of Contents
- [Dataset](#dataset)
- [Code Structure](#code-structure)
- [Data Augmentation](#data-augmentation)
- [Model Training](#model-training)
- [Evaluation](#evaluation)

## Dataset
The project uses four datasets for emotion recognition:
1. **RAVDESS**: (Ryerson Audio-Visual Database of Emotional Speech and Song)
      * This dataset was downloaded from Kaggle (https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)
      * The RAVDESS dataset contains emotional speech and song recordings with a wide range of emotions. It was designed to provide a high-quality resource for emotional speech research.
      * **Speech:** Includes 24 professional actors (12 male, 12 female) who performed 8 emotions: neutral, calm, happy, sad, angry, fearful, disgust, and surprised.
      * **Format:** The audio files are recorded at a sample rate of 48 kHz and are provided in WAV format.
      * **Emotions:** Emotions are labeled numerically (1-8) in the filenames, corresponding to the following: neutral, calm, happy, sad, angry, fearful, disgust, and surprised.
      * **Size:** Approximately 2,880 files (1,440 speech and 1,440 song recordings).
      * **Usage:** Commonly used for emotion recognition, speech synthesis, and affective computing research.

2. **CREMA-D**: (Crowd-sourced Emotional Multimodal Actors Dataset)
    * This dataset was donwload from kaggle (https://www.kaggle.com/datasets/ejlok1/cremad)
    * The CREMA-D dataset consists of emotional speech recordings collected from crowd-sourced actors. The dataset is designed to provide a diverse range of emotional expressions from different demographics.
    * **Speech:** Contains recordings from 91 actors (48 male, 43 female) expressing 6 emotions: happy, sad, angry, disgust, fear, and neutral.
    * **Format:** The audio files are recorded at a sample rate of 22 kHz and are provided in WAV format.
    * **Emotions:** Emotions are labeled with their respective categories in the filenames.
    * **Size:** Over 7,000 files.
    * **Usage:** Used for training and evaluating emotion recognition systems, as well as for studying emotion in speech across different demographics.

   
3. **TESS**: Toronto emotional speech set (TESS)
    * This dataset was downloaded from Kaggle (https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)
    * The TESS dataset was developed to provide a set of emotionally expressive speech samples from a single female speaker. It is useful for studying emotional expression in speech and for training emotion recognition models.
    * **Speech:** Recorded from one actress expressing 7 emotions: angry, disgust, fear, happy, sad, surprise, and neutral.
    * **Format:** Audio files are recorded at a sample rate of 16 kHz and are provided in WAV format.
    * **Emotions:** Emotions are labeled based on the type of expression.
    * **Size:** Approximately 200 files.
    * **Usage:** Typically used for small-scale emotion recognition studies and evaluating emotion expression consistency.


4. **SAVEE**: (Surrey Audio-Visual Expressed Emotion)
   * This dataset was donwloaded from Kaggle (https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee)
   * The SAVEE dataset provides emotional speech recordings from male actors. It is designed to support research in emotion recognition and multimodal emotion analysis.
   * **Speech:** Includes recordings from 4 male actors expressing 7 emotions: angry, disgust, fear, happy, neutral, sad, and surprise.
   * **Format:** Audio files are recorded at a sample rate of 16 kHz and are provided in WAV format.
   * **Emotions:** Emotions are labeled based on the type of expression.
   * **Size:** Approximately 480 files.
   * **Usage:** Utilized for emotion recognition research, especially for studying male emotional expressions in speech.


## Code Structure
1. **Loading Libraries:** Loading required libraries
2.  **Data Loading:** Loads audio files and associated emotions from multiple datasets.
3.  **Feature Extraction:** Extracts various audio features such as zero crossing rate, MFCC, chroma features, and Mel spectrogram.
4.  **Data Augmentation:** Applies noise, stretching, shifting, and pitch shifting to augment the data.
5.  **Model Training:** Builds a CNN model for emotion classification and trains it with the preprocessed data.
6.  **Evaluation:** Evaluates the model using accuracy metrics and generates a confusion matrix and classification report.

# Data Augmentation
The following augmentation techniques are applied to the audio data:
1. **Noise Addition:** Adds Gaussian noise to the audio.
2. **Time Stretching:** Changes the speed of the audio without affecting pitch.
3. **Pitch Shifting:** Alters the pitch of the audio.
4. **Shifting:** Shifts the audio waveform.

## Model Training
The script uses a Convolutional Neural Network (CNN) with the following architecture:
1. **Conv1D Layers:** Multiple convolutional layers for feature extraction.
2. **MaxPooling1D Layers:** Pooling layers to reduce dimensionality.
3. **Dense Layers:** Fully connected layers for classification.
4. **Dropout Layers:** Dropout for regularization.
5. **Model Compilation:** The model is compiled with the Adam optimizer and categorical crossentropy loss function. It is trained for 50 epochs with early stopping based on loss.

# Evaluation
After training, the model's performance is evaluated on the test set. The following metrics are generated:
1. **Accuracy:** Overall accuracy of the model.
2. **Confusion Matrix:** Shows the model's performance across different emotion classes.
3. **Classification Report:** Provides precision, recall, and F1-score for each class.





