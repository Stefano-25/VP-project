# VISION AND PERCEPTION PROJECT

Authors: Paolo Ferretti, Stefano Servillo

# Problem
The problem we've chosen is the deepfake challenge, a phenomenon that has spread especially in recent years. 

A deepfake is a video or a photo of people in which their faces or bodies have been digitally altered to appear as someone else, mostly used to spread false information.

In our project we focused on the case of videos in which only the faces were modified.

# Our solution
The solution we propose is the following:

Once the video to be checked has been chosen, it is passed to our program which divides the video into frames and calculates the optical flow. 

All the optical flows, fused with the RGB frames, are given as input to the model, composed by a pre-trained ResNet18 and a classifier trained for our purpose. 

The model will return its softmax prediction for each pair of frames and, after combining all softmax, it will show the percentage of the video it considers fake.

# Everything you need to run the program
You can find all the code in the notebook and you can run the program in your machine or, as we did, run the program in Google Colab.

The only thing to do is to create an account on [KAGGLE](https://www.kaggle.com/) site, if it has not already been created, and put your username and API key in the third code box. (You can find your username and API key on your account page on kaggle).

It is **NECESSARY** to download the FaceForensics dataset for the training and validation, and our dataset with two videos for the final test, one fake and one original.
