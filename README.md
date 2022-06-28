# VISION AND PERCEPTION PROJECT

Authors: Paolo Ferretti, Stefano Servillo

# Problem
The problem we've chosen is the deepfake challenge, a phenomenon that has spread especially in recent years. A deepfake is a video or a photo of people in which their faces or bodies have been digitally altered to appear as someone else, mostly used to spread false information. In our project we focused on the case of videos in which only the faces was changed.

# Solution
The solution we propose is the following:

Once the video to be checked has been chosen, it is passed to our program which divides the video into frames and calculates the optical flow. All optical flows are given as input to the model, composed by a pre-trained resnet18 model and a classifier trained for out purpose. The model will return its prediction for each pair of frames and, after combining all the results, the program will show the percentage of the video it considers fake.

![schema](/schema.png)

# Program
In the repository you can find the notebook containing everything you need to complete the task.
You only need to add your Kaggle credentials to download the FaceForensic repository and our repository to test 2 casual videos.