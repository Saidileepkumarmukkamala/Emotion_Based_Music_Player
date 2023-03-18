# Emotion_Based_Music_Player
## Introduction:
Emotions are a fundamental aspect of human experience, and music has been shown to have a powerful effect on our emotions. In this project, we will use Convolutional Neural Networks (CNN) to detect emotions from facial expressions and play songs based on the detected emotion.It's a streamlit based web app deployed on cloud.

### Dataset:
To train our CNN model, we will need a dataset of images of people with different emotional states. One of the popular datasets used for this task is the 'FER2013' dataset that contains 35,887 grayscale images of people in seven emotional states: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.

### Preprocessing:
We will start by preprocessing the images before feeding them to the CNN model. We will first resize the images to a standard size (e.g., 48x48 pixels) and then normalize the pixel values between 0 and 1.

### Model Architecture:
For our CNN model, we will use a custom architecture that consists of several convolutional layers, followed by max-pooling layers, and then a few fully connected layers. The final layer will have seven output units, one for each emotional state, with a softmax activation function to predict the probability of each emotion.

### Training:
We will split the dataset into training and validation sets and train the model using the Adam optimizer and categorical cross-entropy loss function. We will also use data augmentation techniques such as rotation, zoom, and horizontal flip to increase the size of the dataset and reduce overfitting.

### Testing:
To test the performance of our model, we will use a separate test set and calculate the accuracy, precision, recall, and F1-score. If the model achieves satisfactory performance, we can move on to the final step.

### Emotion-Based Music Player:
Once the model detects the emotion from the facial expression, it will play a song based on the detected emotion. We can create playlists for different emotions, such as happy, sad, or calm, and select songs that match the mood. We can use a music library such as Spotify or Apple Music and use their APIs to access the songs.

### Conclusion:
In this project, we have developed an emotion-based music player using CNN and played songs based on the detected emotion. This system can potentially enhance the listening experience by selecting songs that match the user's current emotional state. Further improvements can be made by incorporating other sources of data, such as heart rate, skin conductance, or brain activity, to enhance the accuracy of the emotion detection.

## [Learn Visually - Experience the app here](https://emotion-based-music-player.streamlit.app/)
