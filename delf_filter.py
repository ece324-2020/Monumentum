#from delf.delf.python.examples.match_images import get_inliers
#from delf.delf.python.examples.extract_features import create_delf_files
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import time
import platform
from shutil import copy

import sys
# insert at position 1 in the path, as 0 is the path of this file.
sys.path.insert(1, 'delf/delf/python/examples')

print(os.getcwd())

from match_images import get_inliers
from extract_features import create_delf_files

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if (filename!='.DS_Store'):
            img = plt.imread(os.path.join(folder,filename))
            if img is not None:
                images.append(img)
    return images

def add_line_to_txt(line,file_path='delf_parameters/list_images.txt'):
    with open(file_path, 'a') as file:
        file.write(line+'\n')
    return True

def compare_images():
    if platform.system == 'Windows':
        slash = '\\'
    else:
        slash = '/'
    try:
        os.mkdir(os.path.join('delf_parameters','delf_features'))
        os.mkdir(os.path.join('delf_parameters','list_images'))
    except:
        pass
    base = os.path.dirname(os.path.realpath(__file__))
    delf_parameters_path = os.path.join(base,'delf_parameters')
    config_path = os.path.join(delf_parameters_path,'delf_config_example.pbtxt')
    output_dir = os.path.join(delf_parameters_path,'delf_features')
    list_images_path = os.path.join(delf_parameters_path,'list_images')

    dataset_path = os.path.join(base,'dataset')
    dataset_chosen_path = os.path.join(base,'dataset_chosen')
    class_dirs = next(os.walk(dataset_path))[1]

    for path in class_dirs:
        class_path = os.path.join(dataset_path,path)
        class_images = np.array(load_images_from_folder(class_path))

        class_temp = class_path.split(slash)
        print(class_temp)
        print(class_path,'path')
        print(class_images.shape)

        current_class_img_dirs = [os.path.join(os.path.join(class_temp[-2],class_temp[-1]),i) for i in os.listdir(class_path)]
        #current_class_img_dirs = os.join(class_path,os.listdir(class_path))
        #print(current_class_img_dirs)

        #print(os.path.join(list_images_path,path))
        #exit()
        list_img_class_path = os.path.join(list_images_path,path)
        output_dir_class_path = os.path.join(output_dir,path)
        try:
            os.mkdir(list_img_class_path)
        except:
            pass
        #f = open("guru99.txt","w+")
        current_list_images_path = os.path.join(list_img_class_path,'list_images.txt')
        try:
            print(current_list_images_path)
            os.remove(current_list_images_path)
        except:
            pass
        for img_path in current_class_img_dirs:
            if '.DS_Store' not in img_path:
                add_line_to_txt(img_path,file_path = current_list_images_path)
        try:
            shutil.rmtree(output_dir_class_path)
        except:
            pass

        create_delf_files(config_path,current_list_images_path,output_dir_class_path)

    inliers_list=list() #Eventually a list of lists
    #[[('Sample_Image_in_Class_1','Sample_Chosen_Image_in_Class_1',Inliers),('Sample_Image_in_Class_2','Sample_Chosen_Image_in_Class_1',Inliers)],
    for index,path in enumerate(class_dirs):
        if '.DS_Store' in path:
            continue
        else:
            inliers_list+=[[]]
            print(os.path.join(dataset_chosen_path,path))
            for file_path in os.listdir(os.path.join(dataset_chosen_path,path)):
                if '.DS_Store' in file_path:
                    continue
                else:
                    for delf_path in os.listdir(os.path.join(output_dir,path)):
                        if '.DS_Store' in delf_path:
                            continue
                        else:
                            features_1_path = os.path.join(os.path.join(output_dir,path),delf_path)
                            features_2_path = os.path.join(os.path.join(output_dir,path),file_path[:-3]+'delf')
                            image_1_path = None
                            image_2_path = None
                            out_image_path = None
                            features_1_slash = os.path.join(features_1_path.split(slash)[-2],features_1_path.split(slash)[-1])[:-4]+'jpg'
                            features_2_slash = os.path.join(features_2_path.split(slash)[-2],features_2_path.split(slash)[-1])[:-4]+'jpg'
                            if features_1_slash!=features_2_slash:
                                inliers_list[index]+=[(features_1_slash,features_2_slash,get_inliers(features_1_path, features_2_path, image_1_path, image_2_path, out_image_path))]
    threshold = 10
    for index,item in enumerate(inliers_list):
        inliers_list[index] = sorted(item, key=lambda tup: tup[2], reverse=True)

    for index,item in enumerate(inliers_list):
        temp_list = []
        for tuple in item:
            if tuple[2]>threshold:
                temp_list +=[tuple]
        inliers_list[index] = temp_list

    try:
        os.mkdir('dataset_delf_filtered')
    except:
        pass

    print(inliers_list)
    for item in inliers_list:
        slash_vals = item[0][0].split(slash)
        try:
            os.mkdir(os.path.join('dataset_delf_filtered',slash_vals[0]))
        except:
            pass
        for tuple in item:
            copy(os.path.join('dataset',tuple[0]),os.path.join('dataset_delf_filtered',tuple[0]))


compare_images()
exit()



base = os.path.dirname(os.path.realpath(__file__))
delf_parameters_path = os.path.join(base,'delf_parameters')

config_path = os.path.join(delf_parameters_path,'delf_config_example.pbtxt')
list_images_path = os.path.join(delf_parameters_path,'list_images.txt')
output_dir = os.path.join(delf_parameters_path,'test_features')
#create_delf_files(config_path,list_images_path,output_dir)

print(os.getcwd())
print(sys.path)
start = time.time()
create_delf_files(config_path,list_images_path,output_dir)
end=time.time()
print(end-start, 'time')
features_1_path = os.path.join(os.path.join(delf_parameters_path,'test_features'),'image_1.delf')
features_2_path =  os.path.join(os.path.join(delf_parameters_path,'test_features'),'image_2.delf')
image_1_path = os.path.join(os.path.join(delf_parameters_path,'test_images'),'image_1.jpg')
image_2_path = os.path.join(os.path.join(delf_parameters_path,'test_images'),'image_2.jpg')


get_inliers(features_1_path,features_2_path,image_1_path,image_2_path,os.path.join(delf_parameters_path,'out_image.png'))
print(time.time()-end, 'TIMEEEEE')


#compare_images()
