import pandas as pd

label_to_category_data = pd.read_csv('train_label_to_category.csv')
landmark_id_list=[]
clean_data = pd.read_csv('train_clean.csv')
print('CSV Sample Headers:\n')
print(label_to_category_data.head())
print(clean_data.head())
category_vals = label_to_category_data['category'].values

for i,word in enumerate(category_vals):
    if 'Athens' in word or 'athens' in word:
        landmark_id_list+=[i]

print('All Athens IDs:\n',landmark_id_list)
print('ID Count:\n',len(landmark_id_list))

existing_ids=[]
image_tags=[]

for i,id in enumerate(landmark_id_list):
    if id in clean_data['landmark_id'].values:
        existing_ids+=[id]

print('Cross-Validated IDs (IDs that exist):\n',existing_ids)
print('Number of Cross-Validated IDs:\n',len(existing_ids))

landmark_img_pairs = []
for exists in existing_ids:
    print('landmark_id:',exists)
    row = clean_data.loc[clean_data['landmark_id'] == exists]
    images_separated = (row['images'].values[0]).split(' ')
    for img in images_separated:
        landmark_img_pairs.append((row['landmark_id'].values[0],img))

print(landmark_img_pairs)
print(len(landmark_img_pairs))
