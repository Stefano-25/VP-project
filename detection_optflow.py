# Dato il dataset FaceForensics++, voglio calcolare l'optical flow di un video casuale del dataset. Quello che intendo fare è prendere il video, 
# dividerlo in frame, rilevare i volti, tagliare i volti e poi dare in ingresso due frame successivi al mio sistema che calcola l'OF   
import torch
import numpy as np
import cv2
import os
import random


## Number of gpus available
#ngpu = 1
#device = torch.device('cuda:0' if (
    #torch.cuda.is_available() and ngpu > 0) else 'cpu')
def main():

    # Generate random number to extract a random video from the folder deepfakes
    base_dir = r"C:\Users\paolo\OneDrive\Desktop\project_OF\data\original_sequences\youtube\c23\videos"
    file_name = random.choice(os.listdir(base_dir))
    path_to_video = os.path.join(base_dir, file_name)

    # Load the cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Extract the video inside cap and read the first two frames
    cap = cv2.VideoCapture(path_to_video)
    ret, old_frame = cap.read()
    frame1_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(old_frame)
    hsv[..., 1] = 255

    while(1):
        
        if not ret:
            print('No frames grabbed!')
            break
        
        # Detect the faces
        faces = face_cascade.detectMultiScale(frame1_gray, 1.6, 4)
        
        # Draw the rectangle around each face
        for (x, y, w, h) in faces:
            cv2.rectangle(old_frame, (x, y), (x+w, y+h), (255, 0, 0), 4)
        
        # Read new frame
        ret, new_frame = cap.read()
        frame2_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
        
        # Optical flow with Farneback method
        flow = cv2.calcOpticalFlowFarneback(frame1_gray, frame2_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Magnitude and angle of the 2D vectors
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Sets image hue and value according to the optical flow direction
        # and magnitude, then converts HSV to RGB (BGR) color representation
        hsv[..., 0] = ang*180/np.pi/2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Display frames and optical flow
        cv2.imshow('optical flow bgr', bgr)
        cv2.imshow('img', old_frame)
        # cv2.imshow('optical flow hsv', hsv)
        old_frame = new_frame
        frame1_gray = frame2_gray
        
        # Frames are read by intervals of 1 millisecond. The
        # programs breaks out of the while loop when the
        # user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()