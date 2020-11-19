import pandas as pd



def return_wikimedia_list():
    def wiki_list():
        chosen_landmarks = [(38482, 704), (40088, 679), (25093, 663), (162833, 662), (173511, 656), (189907, 621), (190822, 614), (76303, 604), (191292, 600), (109169, 597), (51856, 576), (192931, 572), (41808, 570), (107164, 542), (189811, 540), (101399, 538), (64792, 531), (80177, 527), (27190, 520), (152708, 514), (143710, 508), (31531, 505), (27, 504), (113838, 503), (85633, 502), (19605, 492), (28139, 486), (132969, 485), (115821, 482), (147897, 479), (73300, 477), (107801, 474), (80272, 473),(29794, 463), (171683, 460), (199450, 459), (137203, 456), (39865, 452), (31361, 449), (51272, 437), (165900, 429), (15445, 428), (190956, 428), (98993, 427), (201840, 427), (136302, 426), (70644, 425), (103899, 423), (28641, 418), (180901, 413)]

        landmark_id_list = [tuple[0] for tuple in chosen_landmarks]
        label_to_category_data = pd.read_csv('train_label_to_category.csv')
        clean_data = pd.read_csv('train_clean.csv')

        category_vals = label_to_category_data['category'].values

        return category_vals,landmark_id_list,clean_data

    category_vals,landmark_id_list,clean_data = wiki_list()
    common_ids,image_tags=[],[]

    for i,id in enumerate(landmark_id_list):
        if id in clean_data['landmark_id'].values:
            common_ids+=[id]
    split_category_vals = [i.split(':')[-1].split('_') for i in category_vals[common_ids]]
    extracted_wikimedia_list=[]
    for words in split_category_vals:
        final_string=''
        for index,word in enumerate(words):
            final_string+=str(word)
            if index != len(words) - 1:
                final_string+=' '
        extracted_wikimedia_list+=[final_string]

    final_list = [(landmark_id_list[index],extracted_wikimedia_list[index]) for index in range(len(extracted_wikimedia_list))]
    return final_list

final_list = return_wikimedia_list()
