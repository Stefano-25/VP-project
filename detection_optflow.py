# Dato il dataset FaceForensics++, voglio calcolare l'optical flow di un video casuale del dataset. Quello che intendo fare è prendere il video, 
# dividerlo in frame, rilevare i volti, tagliare i volti e poi dare in ingresso due frame successivi al mio sistema che calcola l'OF   
import numpy as np
import cv2
import os
import random




def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)  
    
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    if (len(faces) == 0):
        return None
    
    return gray[y:y + w, x:x + h]


def compute_optical_flow(prvs, nxt):
    
    # Creates an array filled with zero 
    # with the same dimensions of the frame
    rgb_image = cv2.cvtColor(prvs, cv2.COLOR_GRAY2BGR)
    hsv = np.zeros_like(rgb_image)
    hsv[..., 1] = 255

    # Compute the optical flow
    flow = cv2.calcOpticalFlowFarneback(prvs, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # Magnitude and angle of the 2D vectors
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Sets image hue and value according to the optical flow direction
    # and magnitude, then converts HSV to RGB (BGR) color representation
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return bgr


def main():

    # Generate random number to extract a random video from the folder deepfakes
    base_dir = r"C:\Users\paolo\OneDrive\Desktop\project_OF\data\manipulated_sequences\Face2Face\c23\videos"
    file_name = random.choice(os.listdir(base_dir))
    path_to_video = os.path.join(base_dir, file_name)

    # Extract the video inside cap and read the first two frames
    cap = cv2.VideoCapture(path_to_video)
    ret, old_frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        return

    # Detect the face
    face1 = detect_face(old_frame)

    while(cap.isOpened()):
        
        ret, new_frame = cap.read()
        if not ret:
            print('The video is finished!')
            break
        
        # Read new frame and detect the face
        face2 = detect_face(new_frame)
        
        # Resize the two crops to have the same input dimension in optical flow function
        face1 = cv2.resize(face1, (300, 300), interpolation = cv2.INTER_AREA)
        face2 = cv2.resize(face2, (300, 300), interpolation = cv2.INTER_AREA)

        # Compute the optical flow
        optical_flow = compute_optical_flow(face1, face2)

        # Display frames and optical flow
        cv2.imshow("optical flow", optical_flow)
        cv2.imshow("img", old_frame)
        cv2.imshow("face", face1)
        
        old_frame = new_frame
        face1 = face2
        
        # Frames are read by intervals of 1 millisecond. The
        # programs breaks out of the while loop when the
        # user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()