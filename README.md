# VISION AND PERCEPTION PROJECT

Authors: Paolo Ferretti, Stefano Servillo

# Problem
The problem we have chosen is the deepfake challenge, a phenomenon that has spread especially in recent years. A deepfake is a video or photo of persons in which their face or body has been digitally altered to appear as someone else, mostly used to spread false information. In our project we focused on the case of video in which only the face was changed.

# Solution
The solution we propose is the following:

Once the video to be checked has been chosen, it is passed to our program which will divide the video into frames and calculate the optical flow. All optical flows are given as input to the model, composed by a pre-trained resnet18 model and a classifier trained for out objective. The model will return its result and, after adding all the results, the program will show the precentage of videos it considers fake.

![schema](/schema.png)

# Program
In the repository you can find the notebook containing everything you need to complete the task.
You have only add your credentials of Kaggle to download the FaceForensic repository and our repository to test 2 casual videos.