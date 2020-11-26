import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import time
import platform
from shutil import copy
import pickle

import sys
# insert at position 1 in the path, as 0 is the path of this file.
sys.path.insert(1, 'delf/delf/python/examples')
#print(os.getcwd())
from match_images import get_inliers
from extract_features import create_delf_files

data_directory = 'train_unaugmented'

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
        os.mkdir('dataset_chosen')
    except:
        pass

    base = os.path.dirname(os.path.realpath(__file__))
    delf_parameters_path = os.path.join(base,'delf_parameters')
    config_path = os.path.join(delf_parameters_path,'delf_config_example.pbtxt')
    output_dir = os.path.join(delf_parameters_path,'delf_features')
    list_images_path = os.path.join(delf_parameters_path,'list_images')

    dataset_path = os.path.join(base,data_directory)
    dataset_chosen_path = os.path.join(base,'dataset_chosen')
    #print(os.listdir(dataset_path))
    class_dirs = os.listdir(dataset_path)#next(os.walk(dataset_path))[1]
    '''
    start = time.time()
    for path in class_dirs:
        anew = time.time()
        class_path = os.path.join(dataset_path,path)
        if '.DS_Store' in class_path:
            continue
        class_images = np.array(load_images_from_folder(class_path))

        class_temp = class_path.split(os.sep)

        #print(class_temp)
        #print(class_path,'path')
        #print(class_images.shape)


        current_class_img_dirs = [os.path.join(os.path.join(class_temp[-2],class_temp[-1]),i) for i in os.listdir(class_path)]
        #current_class_img_dirs = os.join(class_path,os.listdir(class_path))
        ##print(current_class_img_dirs)

        ##print(os.path.join(list_images_path,path))
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
            #print(current_list_images_path)
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
        print('Time taken to process class {} is {}'.format(str(path),str(time.time()-anew)))
        print('Cumulative time taken so far is {}'.format(str(time.time()-start)))


    #end=time.time()

    #print(end-start,"FILTER TIME")
    '''
    try:
        os.mkdir('dataset_delf_filtered')
    except:
        pass
    inliers_list=list() #Eventually a list of lists
    #[[('Sample_Image_in_Class_1','Sample_Chosen_Image_in_Class_1',Inliers),('Sample_Image_in_Class_2','Sample_Chosen_Image_in_Class_1',Inliers)],
    threshold = 20
    start = time.time()
    print(class_dirs)

    def check_in_path(in_path):
        done_classes = ['15445','101399','103899','107164','107801','109169','113838','115821','132969','162833','173511','189907','190822','191292']
        for i in done_classes:
            if i in in_path:
                return True
        return False

    for index,path in enumerate(class_dirs):
        try:
            #print(path)
            os.mkdir(os.path.join('dataset_delf_filtered',path))
        except:
            pass
        #os.mkdir(os.path.join('dataset_delf_filtered',path))
        anew = time.time()
        if '.DS_Store' in path or check_in_path(path) == True:
            inliers_list+=[[]]
            continue
        else:
            inliers_list+=[[]] #index = 1 #inliers_list has only 1 item, so inliers_list[1] error
            #print(os.path.join(dataset_chosen_path,path))
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
                            features_1_slash = os.path.join(features_1_path.split(os.sep)[-2],features_1_path.split(os.sep)[-1])[:-4]+'jpg'
                            features_2_slash = os.path.join(features_2_path.split(os.sep)[-2],features_2_path.split(os.sep)[-1])[:-4]+'jpg'
                            #output_image = '{}.jpg'.format(counter)
                            #print(output_image)
                            #print('dataset_chosen'+os.sep+features_1_slash,data_directory+os.sep+features_2_slash)
                            #inliers = get_inliers(features_1_path, features_2_path, data_directory+os.sep+features_1_slash, 'dataset_chosen'+os.sep+features_2_slash, output_image)
                            #try:
                            print(features_1_slash,features_2_slash)
                            inliers = get_inliers(features_1_path, features_2_path, None, None, None)
                            #except:
                            #    inliers = 0
                            if features_1_slash!=features_2_slash and inliers>threshold:
                                copy(os.path.join(data_directory,features_1_slash),os.path.join('dataset_delf_filtered',features_1_slash))

                            inliers_list[index]+=[(features_1_slash,features_2_slash,inliers)]

        print('Time taken to add inliers for class {} is {}'.format(str(path),str(time.time()-anew)))
        print('Cumulative time to do inliers is {}'.format(str(time.time()-start)))

    with open('delf_filter_inliers_list', 'wb') as fp:
        pickle.dump(inliers_list, fp)

    '''
    #for index,item in enumerate(inliers_list):
    #    inliers_list[index] = sorted(item, key=lambda tup: tup[2], reverse=True)

    for index,item in enumerate(inliers_list):
        temp_list = []
        for tuple in item:
            if tuple[2]>threshold:
                temp_list +=[tuple]
        inliers_list[index] = temp_list

    for item in inliers_list:
        slash_vals = item[0][0].split(slash)
        try:
            os.mkdir(os.path.join('dataset_delf_filtered',slash_vals[0]))
        except:
            pass
        for tuple in item:
    '''
compare_images()
#features_1_path = 'delf_parameters' + os.sep + 'delf_features' + os.sep + '115821' + os.sep + '0b7433d632175d3d.delf'
#features_2_path = 'delf_parameters' + os.sep + 'delf_features' + os.sep + '115821' + os.sep + 'a9a27ffc14adee02.delf'
#img_1_path = 'train_unaugmented' + os.sep + '115821' + os.sep + '0b7433d632175d3d.jpg'
#img_2_path = 'train_unaugmented' + os.sep + '115821' + os.sep + 'a9a27ffc14adee02.jpg'

#inliers = get_inliers(features_1_path, features_2_path, None, None, None)
exit()
#fc1138f39bc5b379.jpg
#2fe6b86c91fcc98f.jpg
'''
base = os.path.dirname(os.path.realpath(__file__))
delf_parameters_path = os.path.join(base,'delf_parameters')

config_path = os.path.join(delf_parameters_path,'delf_config_example.pbtxt')
list_images_path = os.path.join(delf_parameters_path,'list_images.txt')
output_dir = os.path.join(delf_parameters_path,'test_features')
#create_delf_files(config_path,list_images_path,output_dir)

#print(os.getcwd())
#print(sys.path)
start = time.time()
create_delf_files(config_path,list_images_path,output_dir)
end=time.time()
#print(end-start, 'time')
features_1_path = os.path.join(os.path.join(delf_parameters_path,'test_features'),'image_1.delf')
features_2_path =  os.path.join(os.path.join(delf_parameters_path,'test_features'),'image_2.delf')
image_1_path = os.path.join(os.path.join(delf_parameters_path,'test_images'),'image_1.jpg')
image_2_path = os.path.join(os.path.join(delf_parameters_path,'test_images'),'image_2.jpg')


get_inliers(features_1_path,features_2_path,image_1_path,image_2_path,os.path.join(delf_parameters_path,'out_image.jpg'))
#print(time.time()-end, 'TIMEEEEE')


#compare_images()'''
