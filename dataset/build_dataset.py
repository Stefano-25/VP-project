# USAGE
# python build_dataset.py
# import necessary packages
from imutils import paths
import numpy as np
import shutil
import os
import config



def copy_videos(videoPaths, folder):
	
    # check if the destination folder exists and if not create it
	if not os.path.exists(folder):
		os.makedirs(folder)
	
    # loop over the image paths
	for path in videoPaths:
		
        # grab image name and its label from the path and create
		# a placeholder corresponding to the separate label folder
		videoName = path.split(os.path.sep)[-1]
		label = path.split(os.path.sep)[-2]
		labelFolder = os.path.join(folder, label)
		
        # check to see if the label folder exists and if not create it
		if not os.path.exists(labelFolder):
			os.makedirs(labelFolder)
		
        # construct the destination image path and copy the current
		# image to it
		destination = os.path.join(labelFolder, videoName)
		shutil.copy(path, destination)


# load all the image paths and randomly shuffle them
print("[INFO] loading video paths...")
videoPaths = list(paths.list_files(config.DATASET_PATH))
np.random.shuffle(videoPaths)

# generate training and validation paths
valPathsLen = int(len(videoPaths) * config.VAL_SPLIT)
trainPathsLen = len(videoPaths) - valPathsLen
trainPaths = videoPaths[:trainPathsLen]
valPaths = videoPaths[trainPathsLen:]

# copy the training and validation images to their respective
# directories
print("[INFO] copying training and validation videos...")
copy_videos(trainPaths, config.TRAIN)
copy_videos(valPaths, config.VALIDATION)