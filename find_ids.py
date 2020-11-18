import pandas as pd

label_to_category_data = pd.read_csv('train_label_to_category.csv')
landmark_id_list=[]
clean_data = pd.read_csv('train_clean.csv')
print(label_to_category_data.head())
print(clean_data.head())
category_vals = label_to_category_data['category'].values

for i,word in enumerate(category_vals):
    if 'Athens' in word or 'athens' in word:
        landmark_id_list+=[i]

print(landmark_id_list)
print(len(landmark_id_list))

common_ids=[]
image_tags=[]
for i,id in enumerate(landmark_id_list):
    if id in clean_data['landmark_id'].values:
        common_ids+=[id]

print(common_ids)
print(len(common_ids))
