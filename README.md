# VISION AND PERCEPTION PROJECT

Authors: Paolo Ferretti, Stefano Servillo

# Problem
The problem we've chosen is the deepfake challenge, a phenomenon that has spread especially in recent years. 

A Deepfake is a digitally manipulated image or video in which a person is replaced with someone else’s likeness, mostly used to spread false information and for frauds.

In our project we focus on the case of videos in which only the faces are modified and we expect to detect the synthetic anomalies training the model with the video's frames and its optical flow.

# Our solution
We present below the main steps of our solution:

1) To focus on the modified parts and to reduce the computation time, we take all the videos and we first crop the faces from each frame using the Haar feature-based Cascade classifier, a face detection algorithm.

2) Next, we compute the optical flow. We use the Farneback optical flow, which is a dense optical flow method which gave us a 2D array with the two flow vectors. Later we visualize the direction of the flow by hue and the magnitude of the flow by value of HSV color representation. This is then converted into an RGB image.

3) Subsequently, we combine each optical flow with the related frames. We compute a 50% weighted average on the frames, using the cv2 AddWeighted function, to obtain a fusion/blending of them.
Then we use this fusion to do a second weighted average with the optical flow, in this case we used 70% from the RGB fusion and 30% from the optical flow.

4) The next step is to train, validate and test our pre-trained ResNet model, where the last fully connected layer is replaced with a new fully connected layer with two output features.
We take 960 videos, divide them 70% for training, 20% for validation and the remaining 10% for testing. On the training set we apply data augmentation, while for validation and test set, we do not apply any transformation. Obviously, we apply the transformations to normalize and to turn the frames into PyTorch tensors to all the sets.
We train and validate the model for 30 epochs. The dataset is divided in batch with dimension 128. In the training phase we use the CrossEntropyLoss as loss function, the SGD as optimizer with learning rate 0.001 and momentum 0.9 and the STEPLR scheduler.

5) At the end, we test the model with the 96 videos. 

Classification: we use the SoftMax to compute the accuracy of our model. What we do, is taking the SoftMax for each frame of the video. Since we have many frames for a single video, we compute an average of the SoftMax obtained for each frame.

# Everything you need to run the program
You can find all the code in the notebook and you can run the program in your machine or, as we did, run the program in Google Colab.

In the process we use as dataset to train, validate and test our model, the FaceForensics++ dataset. We download it from Kaggle, and it contains of 1000 original sequences  that have been manipulated with four automated face manipulation methods: Deepfakes, Face2Face, FaceSwap and NeuralTextures.  In our project, we consider only the DeepFake videos, and we take in total 960 videos, half counterfeit and half original.
It is **NECESSARY** to download the FaceForensics++ dataset for the training, validation and test.

The only thing to do is to create an account on the [KAGGLE](https://www.kaggle.com/) site, and put your username and API key in the code box. (You can find your username and API key on your account page on kaggle).
