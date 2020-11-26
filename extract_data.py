#from id_preprocessing import get_existing_landmarks
import pandas as pd
import os
import shutil
def get_images():
    clean_data = pd.read_csv('train_clean.csv')
    #landmarkid_freq_pairs = get_existing_landmarks()
    landmarkid_freq_pairs = [(38482, 704), (40088, 679), (25093, 663), (162833, 662), (173511, 656),
    (189907, 621), (190822, 614), (76303, 604), (191292, 600), (109169, 597), (51856, 576), (192931, 572),
    (41808, 570), (107164, 542), (189811, 540), (101399, 538), (64792, 531), (80177, 527), (27190, 520),
    (152708, 514), (143710, 508), (31531, 505), (27, 504), (113838, 503), (85633, 502), (19605, 492),(28139, 486),
    (132969, 485), (115821, 482), (147897, 479), (73300, 477), (107801, 474), (80272, 473), (29794, 463),
    (171683, 460), (199450, 459), (137203, 456), (39865, 452), (31361, 449), (51272, 437), (165900, 429),
    (15445, 428), (190956, 428), (98993, 427), (201840, 427), (136302, 426), (70644, 425), (103899, 423),
    (28641, 418), (180901, 413)]
    landmark_id_img_pairs = []
    for pair in landmarkid_freq_pairs:
        row = clean_data.loc[clean_data['landmark_id'] == pair[0]]
        img_id_string = row['images'].values[0].split(' ')
        # append tuple of (landmark_id, list of image ids) for each landmark
        landmark_id_img_pairs.append((row['landmark_id'].values[0],img_id_string))
    return landmark_id_img_pairs


def main_extract(complete_data_dir,extracted_data_dir):
    '''
    complete_data_dir : the directory of the full 98GB dataset from Kaggle
    '''
    # Get list of tuples where each tuple is (landmark_id, list of image_id (strings) corresponding to landmark_id)
    landmark_image_data = get_images()
    img_extension = '.jpg'
    # Make the directory folder
    if not os.path.exists(extracted_data_dir):
        os.mkdir(extracted_data_dir)

    for landmark_tuple in landmark_image_data:

        subpath = extracted_data_dir+os.sep+str(landmark_tuple[0])
        if not os.path.exists(subpath):
            os.mkdir(subpath)
            print('Created Directory:',subpath)

        print('Transferring Class:',landmark_tuple[0])
        # With folder made, transfer corresponding images via image_id
        for img_id in landmark_tuple[1]:
            sub1 = img_id[0]
            sub2 = img_id[1]
            sub3 = img_id[2]
            # path to the original image in the complete dataset
            source_path = complete_data_dir + os.sep + sub1 + os.sep + sub2 + os.sep + sub3 + os.sep + img_id + img_extension
            shutil.copy(source_path,subpath)

def get_subset():
    # Get subset of data
    landmarkid_freq_pairs = [(38482, 704), (40088, 679), (25093, 663), (162833, 662), (173511, 656),
        (189907, 621), (190822, 614), (76303, 604), (191292, 600), (109169, 597), (51856, 576), (192931, 572),
        (41808, 570), (107164, 542), (189811, 540), (101399, 538), (64792, 531), (80177, 527), (27190, 520),
        (152708, 514), (143710, 508), (31531, 505), (27, 504), (113838, 503), (85633, 502), (19605, 492),(28139, 486),
        (132969, 485), (115821, 482), (147897, 479), (73300, 477), (107801, 474), (80272, 473), (29794, 463),
        (171683, 460), (199450, 459), (137203, 456), (39865, 452), (31361, 449), (51272, 437), (165900, 429),
        (15445, 428), (190956, 428), (98993, 427), (201840, 427), (136302, 426), (70644, 425), (103899, 423),
        (28641, 418), (180901, 413)]
    folders = ['train','test','val']
    for folder in folders:
        for landmark in landmarkid_freq_pairs:
            orig_path = 'extracted_data_dir_split'+os.sep+folder+os.sep+str(landmark[0])
            subset_path = 'extracted_data_dir_split_subset'+os.sep+folder+os.sep+str(landmark[0])
            if not os.path.exists(subset_path):
                os.mkdir(subset_path)
            images = os.listdir(orig_path)
            if folder == 'train':
                counter = 100
            else:
                counter = 11
            for i in range(counter):
                shutil.copy(orig_path+os.sep+images[i],subset_path)

def remerge_data_set():
    landmarkid_freq_pairs = [(38482, 704), (40088, 679), (25093, 663), (162833, 662), (173511, 656),
        (189907, 621), (190822, 614), (76303, 604), (191292, 600), (109169, 597), (51856, 576), (192931, 572),
        (41808, 570), (107164, 542), (189811, 540), (101399, 538), (64792, 531), (80177, 527), (27190, 520),
        (152708, 514), (143710, 508), (31531, 505), (27, 504), (113838, 503), (85633, 502), (19605, 492),(28139, 486),
        (132969, 485), (115821, 482), (147897, 479), (73300, 477), (107801, 474), (80272, 473), (29794, 463),
        (171683, 460), (199450, 459), (137203, 456), (39865, 452), (31361, 449), (51272, 437), (165900, 429),
        (15445, 428), (190956, 428), (98993, 427), (201840, 427), (136302, 426), (70644, 425), (103899, 423),
        (28641, 418), (180901, 413)]
    folders = ['train','test','val']
    merge_path = 'extracted_data_unsplit_subset'
    split_path = 'extracted_data_dir_split_subset'
    for folder in folders:
        for landmark in landmarkid_freq_pairs:
            split_path_temp = split_path+os.sep+folder+os.sep+str(landmark[0])
            merge_path_temp = merge_path + os.sep + str(landmark[0])
            if not os.path.exists(merge_path_temp):
                os.mkdir(merge_path_temp)
            images = os.listdir(split_path_temp)
            for i in images:
                shutil.copy(split_path_temp+os.sep+i,merge_path_temp)
                #print('Copied {} to {}'.format(split_path_temp+os.sep+i,merge_path_temp))






if __name__ == '__main__':
    #get_subset()
    remerge_data_set()
