import splitfolders
import os
splitfolders.ratio('data_main'+os.sep+'dataset_delf_filtered_augmented', output="dataset_delf_filtered_augmented_split", seed=1337, ratio=(.8, 0.1,0.1))
