import pandas as pd
import numpy as np
import os

def max_val_accuracy(file_path):
    pass

def accuracy_getter(folder_path):
    if os.listdir(folder_path)==[]:
        return True
    for file in os.listdir(folder_path):
        if file[-4:]=='xlsx':
            file_path = os.path.join(folder_path,file)
    df = pd.read_excel(file_path)
    epochs = int(df.columns[-1].split('_')[-1])+1
    val_columns = ['valid_accuracy_epoch_{}'.format(i) for i in range(0,epochs,1)]
    test_columns = ['test_accuracy_epoch_{}'.format(i) for i in range(0,epochs,1)]

    val_accuracy_values = df[val_columns].values.squeeze()
    test_accuracy_values = df[test_columns].values.squeeze()
    return val_accuracy_values, test_accuracy_values

def stats_getter(folder_path):
    val_accuracy_values, test_accuracy_values = accuracy_getter(folder_path)

    max_val_accuracy = max(val_accuracy_values)
    max_test_accuracy = max(test_accuracy_values)

    end_val_accuracy = val_accuracy_values[-1]
    end_test_accuracy = test_accuracy_values[-1]

    stats = folder_path.split(os.sep)[-1]
    innovation_type = folder_path.split(os.sep)[-2]
    stats = [i.split('=')[-1] for i in stats.split('_')]
    print('Folder is called {}'.format(innovation_type))
    print('-----------------------------------------------------')
    print('It uses the {} model, with {} optimizer, {} LR, {} batch size and {} epochs'.format(stats[0],stats[1],stats[2],stats[3],stats[-1]))
    print('Final test accuracy is {}'.format(end_test_accuracy))
    print('Final val accuracy is {}'.format(end_val_accuracy))
    print('Max test accuracy is {}'.format(max_test_accuracy))
    print('Max val accuracy is {}'.format(max_val_accuracy))
    print('-----------------------------------------------------')
    print('\n')

if __name__=='__main__':
    '''folder_path = '/Users/Admin/Documents/GitHub/Monumentum/Training_Runs/ResNext_Best_Run/Model=ResNext101_Optim=Adam_LR=0.001_batchsize=16_epochs=150_momentum=0'
    for i in os.listdir('All_Runs'):
        file_path = os.path.join('All_Runs',i)
        if os.path.isdir(file_path)==True:
            for j in os.listdir(file_path):
                if os.path.isdir(os.path.join(file_path,j))==True:
                    stats_getter(os.path.join(file_path,j))'''
    best_baseline_run = ''
    best_vgg_run = '/Users/Admin/Documents/GitHub/Monumentum/All_Runs/VGG_Runs/Model=VGG16_Optim=Adam_LR=0.001_batchsize=16_epochs=100_momentum=0'
    best_resnet_run = ''
    best_original_resnext_run = ''
    best_resnext_run = '/Users/Admin/Documents/GitHub/Monumentum/All_Runs/ResNext_Best_Run_96/Model=ResNext101_Optim=Adam_LR=0.0005_batchsize=12_epochs=150_momentum=0'
    print(accuracy_getter(best_resnext_run)[1][:100])
    label_values = ['Baseline Model', 'VGG Model', 'ResNet34 Model', 'Original ResNext Run', 'Final ResNext Model']
