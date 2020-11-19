import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import statistics

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if (filename!='.DS_Store'):
            img = plt.imread(os.path.join(folder,filename))
            if img is not None:
                images.append(img)
    return images

def augment_images(image_list):
    new_images=[]
    for image in image_list:
        new_image = (np.flipud(image))
        new_images+=[new_image]
    return new_images

def get_median_data(set_path):
    class_dirs = next(os.walk(set_path))[1]
    lengths = []
    for path in class_dirs:
        original_images = np.array(load_images_from_folder(os.path.join(set_path,path)))
        lengths = lengths + [original_images.shape[0]]
    return statistics.median(lengths)


def expand_class_images(images, target_length):
    num_required_images = (target_length-images.shape[0])
    #print(num_required_images)
    augmented_images = augment_images(images[0:num_required_images])
    #print(images.shape)
    new_images = np.append(images,augmented_images,axis=0)
    #print(new_images.shape)
    return new_images

def shrink_class_images(images, target_length):
    return images[0:target_length]

#e.g. path_original=train/valid/test
def create_augment_folder(set_path, dataset_path, set_dir):
    new_set_dir = os.path.join(dataset_path,set_dir+'_augmented')
    try:
        os.mkdir(new_set_dir)
    except OSError as error:
        print(error)

    #data_lengths_median = 100
    data_lengths_median = int(get_median_data(set_path))
    print("Median of Class Images:",data_lengths_median)
    new_set_images=[]
    class_dirs = next(os.walk(set_path))[1]

    for path in class_dirs:
        print("---Current Class:", path)
        class_path = os.path.join(set_path,path)
        class_images = np.array(load_images_from_folder(class_path))
        new_class_images=[]
        if (class_images.shape[0]>=data_lengths_median):
            print("Removing Images from Class")
            new_class_images = shrink_class_images(class_images,data_lengths_median)
        else:
            print("Adding Images to Class")
            new_class_images = expand_class_images(class_images,data_lengths_median)

        print("OLD SIZE:",class_images.shape[0])
        print("NEW SIZE:",new_class_images.shape[0])
        new_class_dir = os.path.join(new_set_dir, path)
        try:
            os.mkdir(new_class_dir)
        except OSError as error:
            print(error)

        for i,image in enumerate(new_class_images):
            plt.imsave(os.path.join(new_class_dir,str(i)+'.png'),image)

def create_augmented_folders():
    base = os.path.dirname(os.path.realpath(__file__))
    dataset_path = os.path.join(base,'dataset')
    train_dir='train'
    val_dir='val'
    test_dir='test'

    train_path = os.path.join(dataset_path,train_dir)
    val_path = os.path.join(dataset_path,val_dir)
    test_path = os.path.join(dataset_path,test_dir)

    create_augment_folder(train_path, dataset_path, train_dir)
    create_augment_folder(val_path, dataset_path, val_dir)
    create_augment_folder(test_path, dataset_path, test_dir)

create_augmented_folders()
