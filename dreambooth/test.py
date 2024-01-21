import os
import cv2
import shutil
from tqdm import tqdm


name_list = os.listdir('results_good')
name_list = [f for f in name_list if '.jpg' in f]

for idx, name in enumerate(name_list):
    image = cv2.imread('results_good/' + name)
    new_folder = 'results_split/' + name.split('.')[0]
    suffix = '-' + name.split('-')[-1]
    for i in range(4):
        for j in range(4):
            image_crop = image[512*i:512*(i+1), 512*j:512*(j+1), :]
            cv2.imwrite('results_split/' + name.replace(suffix, 
                                        '-{}-{}.jpg'.format(i, j)), image_crop)


    
    
'''
for folder in os.listdir('results_split'):
    sub_folder = 'results_split/' + folder
    suffix = '-' + folder.split('-')[-1]
    name_list = os.listdir(sub_folder)
    name_list = [f for f in name_list if '.jpg' in f]
    #if len(name_list) > 0:
    old_path = sub_folder + '/' + name_list[0]
    new_path = 'results_split/' + folder + '.jpg'
    shutil.copyfile(old_path, new_path)
'''
    

    