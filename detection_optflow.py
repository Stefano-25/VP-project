# Dato il dataset FaceForensics++, voglio calcolare l'optical flow di un video casuale del dataset. Quello che intendo fare è prendere il video, 
# dividerlo in frame, rilevare i volti, tagliare i volti e poi dare in ingresso due frame successivi al mio sistema che calcola l'OF   
import torch
import numpy as np
import cv2
import os
import random


def main():
    
    ## Number of gpus available
    #ngpu = 1
    #device = torch.device('cuda:0' if (
        #torch.cuda.is_available() and ngpu > 0) else 'cpu')

    # Take random file from the data folder
    base_dir = r"C:\Users\paolo\OneDrive\Desktop\project_OF\data\original_sequences\youtube\c23\videos"
    file_name = random.choice(os.listdir(base_dir))
    path_to_video = os.path.join(base_dir, file_name)

    # Extract the video inside cap and read the first two frames.
    # Extract the video and the first frame, convert to gray scale.
    cap = cv2.VideoCapture(path_to_video)
    ret, old_frame = cap.read()
    frame1_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    
    # Creates an array filled with zero 
    # with the same dimensions of the frame
    hsv = np.zeros_like(old_frame)
    hsv[..., 1] = 255

    # read second frame and convert to gray scale
    ret, new_frame = cap.read()
    if not ret:
            print('No frames grabbed!')
            quit()
    frame2_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

    ## optical flow with Farneback method
    flow = cv2.calcOpticalFlowFarneback(frame1_gray, frame2_gray, None, 0.5, 3, 3, 1, 5, 1.2, 0)
    
    # magnitude and angle of the 2D vectors
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Sets image hue and value according to the optical flow direction
    # and magnitude, then converts HSV to RGB (BGR) color representation
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    ## Display frames and optical flow
    # cv2.imshow('optical flow', hsv)
    cv2.imshow('optical flow', bgr)
    cv2.imshow('frame1', frame1_gray)
    cv2.imshow('frame2', frame2_gray)
    cv2.waitKey()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
