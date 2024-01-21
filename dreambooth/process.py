import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm
from face_sdk.face_sdk import get_keypoints

name_list = os.listdir('dataset_body')
name_list = [f for f in name_list if '.jpg' in f]
name_list.sort()
for name in tqdm(name_list):
	old_path = os.path.join('dataset_body', name)
	image = cv2.imread(old_path)
	try:
		landmarks = get_keypoints(image)
		x0, x1 = np.min(landmarks[:, 0]), np.max(landmarks[:, 0])
		y0, y1 = np.min(landmarks[:, 1]), np.max(landmarks[:, 1])
		x_mean, y_mean = (x0 + x1) / 2, (y0 + y1) / 2
		size = x_mean - x0
		x00, x01 = int(x_mean-1.5*size), int(x_mean+1.5*size)
		y00, y01 = int(y_mean-1.8*size), int(y_mean+1.2*size)
		new_path = "dataset_body/processed_{}_{}_{}_{}.jpg".format(x00, x01, y00, y01)
		shutil.copyfile(old_path, new_path)
	except:
		pass