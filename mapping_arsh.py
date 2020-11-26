import shutil
import os
from shutil import copy
last_10_ids = [199450,191292,190822,189907,173511,162833,132969,115821,11383]

corresponding_mappings = [
                              ['0ec597308dd16503','1','10'],
                              ['0','104d00c291aa01a3','112'],
                              ['0','0a5f63b89bfc6b15','1'],
                              ['0','11','12'],
                              ['0','0f6920183dab3ed5','04ec626cab88d089'],
                              ['0','1f90b6018bef9cb2','07ede7d2eea33959'],
                              ['0ebd7c853937f43d','2ccca23b2a27b03b','111'],
                              ['0f95d5e6838347eb','1','1e7db2b9afd2911b'],
                              ['0d480585d123c50a','2e83fbcb16b8385e','11'],
                              ['1a5cfd80a6afbc6e','1cc641d33ca88387','16c0954ee3a2ffbf'],
                              ['109','0fc0e9b72a531fa0','3ee0bc57a0f6d1c5'],
                              ['0dea991aa125afbe','18','114'],
                              ['10','12','16'],
                              ['106','120','125'],
                              ['0ffb0cd39b8589b2','2f6f1aff6dff51ea','12']
                         ]

my_dict = {}
for index,item in enumerate(last_10_ids):
    my_dict[str(item)] = corresponding_mappings[index]

unsplit_directory_path = 'train_unaugmented'
output_directory = 'dataset_chosen' #add your directory name here
'''
try:
    os.mkdir(output_directory)
except:
    pass

for key in my_dict:
    for id in my_dict[key]:
        try:
            os.mkdir(os.path.join(output_directory,key))
        except:
            pass
        copy(os.path.join(os.path.join(unsplit_directory_path,key),id+'.jpg'),os.path.join(os.path.join(output_directory,key),id+'.jpg'))'''

images_dont_exist = []
for key in my_dict:
    for image in my_dict[key]:
        if os.path.isfile(unsplit_directory_path + os.sep + key + os.sep + image + '.jpg')==False:
            images_dont_exist+=[unsplit_directory_path + os.sep + key + os.sep + image + '.jpg']
print(images_dont_exist)
