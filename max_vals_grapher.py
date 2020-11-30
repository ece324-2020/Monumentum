import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter

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
    test_loss_columns = ['test_loss_epoch_{}'.format(i) for i in range(0,epochs,1)]
    val_accuracy_values = df[val_columns].values.squeeze()
    test_accuracy_values = df[test_columns].values.squeeze()
    test_loss_values = df[test_loss_columns].values.squeeze()
    return val_accuracy_values, test_accuracy_values, test_loss_values

def output_scaler(output):
    x_vals = [i+1 for i in range(len(output))]
    new_x_vals = np.linspace(1,len(output),150)
    output = np.interp(new_x_vals,x_vals,output)
    return output

def stats_getter(folder_path):
    val_accuracy_values, test_accuracy_values, test_loss_values = accuracy_getter(folder_path)

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
    folder_path = '/Users/Admin/Documents/GitHub/Monumentum/Training_Runs/ResNext_Best_Run/Model=ResNext101_Optim=Adam_LR=0.001_batchsize=16_epochs=150_momentum=0'
    for i in os.listdir('All_Runs'):
        file_path = os.path.join('All_Runs',i)
        if os.path.isdir(file_path)==True:
            for j in os.listdir(file_path):
                if os.path.isdir(os.path.join(file_path,j))==True:
                    stats_getter(os.path.join(file_path,j))
    best_baseline_run = ''
    best_vgg_run = '/Users/Admin/Documents/GitHub/Monumentum/All_Runs/Best_VGG_Run/Model=VGG16_Optim=Adam_LR=0.001_batchsize=32_epochs=50_momentum=0'
    best_resnet_run = '/Users/Admin/Documents/GitHub/Monumentum/All_Runs/Best_ResNet_Run/Model=ResNet34_Optim=SGD_LR=0.001_batchsize=32_epochs=100_momentum=0'
    best_original_resnext_run = '/Users/Admin/Documents/GitHub/Monumentum/All_Runs/ResNext_Best_Run_2/Model=ResNext101_Optim=Adam_LR=0.001_batchsize=16_epochs=150_momentum=0'
    best_resnext_run = '/Users/Admin/Documents/GitHub/Monumentum/All_Runs/ResNext_Best_Run_96/Model=ResNext101_Optim=Adam_LR=0.0005_batchsize=12_epochs=150_momentum=0'
    folder_paths = [best_baseline_run,best_vgg_run,best_resnet_run,best_original_resnext_run,best_resnext_run]
    label_values = ['Baseline', 'VGG', 'ResNet34', 'Original ResNext', 'Modified ResNext']
    x_vals = [i+1 for i in range(len(accuracy_getter(best_resnext_run)[1]))]
    fig = plt.plot()
    for index,path in enumerate(folder_paths):
        if label_values[index]=='Baseline':
            test_accuracy_values = output_scaler([0.1325, 0.2675, 0.30125, 0.35875, 0.3375, 0.355, 0.42, 0.43125, 0.46125, 0.46625, 0.4525, 0.47625, 0.5025, 0.5325, 0.5225, 0.55375, 0.56375, 0.52875, 0.52375, 0.57, 0.5925, 0.57375, 0.59375, 0.55375, 0.595, 0.59875, 0.5775, 0.5825, 0.58875, 0.58875, 0.605, 0.605, 0.5725, 0.60875, 0.61125, 0.6175, 0.61375, 0.5925, 0.59875, 0.61625, 0.62, 0.62625, 0.62125])
        else:
            test_accuracy_values = accuracy_getter(path)[1]
            if label_values[index] == 'VGG' or label_values[index] == 'ResNet34':
                test_accuracy_values = output_scaler(test_accuracy_values)
        plt.plot(x_vals,test_accuracy_values,label = label_values[index])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim(0.5,1)
    plt.title('Test Accuracy Plots for Different Models')
    plt.legend(loc='lower left',fontsize=9)
    plt.savefig('Different_Models.jpg',bbox_inches='tight',dpi = 700,figsize = (15,4.34))
    plt.clf()

    fig = plt.plot()
    colors = ['tab:red','tab:orange','tab:green','tab:red','tab:purple']
    for index,path in enumerate(folder_paths):
        if label_values[index]=='Baseline':
            continue
        else:
            test_loss_values = accuracy_getter(path)[2]
            if label_values[index] == 'VGG' or label_values[index] == 'ResNet34':
                test_loss_values = output_scaler(test_loss_values)
        plt.plot(x_vals,test_loss_values,label=label_values[index],color=colors[index])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.ylim(0,1.2)
    plt.title('Test Loss Plots for Different Models')
    plt.legend(loc='lower right',fontsize=9)
    plt.savefig('Different_Models_Loss.jpg',bbox_inches='tight',dpi = 700,figsize = (15,4.34))
    plt.clf()

    delf = best_resnext_run
    no_delf_augmented = '/Users/Admin/Documents/GitHub/Monumentum/All_Runs/Unsplit_Augmented_NoDelf'
    no_delf_no_augment = '/Users/Admin/Documents/GitHub/Monumentum/All_Runs/Unsplit_Unaugmented_NoDelf/Model=ResNext101_Optim=Adam_LR=0.001_batchsize=16_epochs=150_momentum=0'
    folder_path = [delf, no_delf_augmented, no_delf_no_augment]
    label_values = ['DELF','No DELF, Augmented', 'No DELF, Unaugmented']

    for index, path in enumerate(folder_path):
        test_accuracy_values = accuracy_getter(path)[1]
        plt.plot(x_vals,test_accuracy_values,label = label_values[index])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Plots for Different Kinds of Dataset Techniques')
    plt.legend(loc='lower right')
    plt.savefig('Different_Data_Techniques.jpg',bbox_inches='tight',dpi = 700,figsize = (15,4.34))
    plt.clf()#

    fig = plt.plot()
    folder_paths = [best_baseline_run,best_vgg_run,best_resnet_run,best_original_resnext_run,best_resnext_run]
    label_values = ['Baseline', 'VGG', 'ResNet34', 'Original ResNext', 'Modified ResNext']
    colors = ['tab:red','tab:orange','tab:red','tab:red','tab:purple']
    for index,path in enumerate(folder_paths):
        if label_values[index]=='Baseline':
            test_accuracy_values = output_scaler([0.1325, 0.2675, 0.30125, 0.35875, 0.3375, 0.355, 0.42, 0.43125, 0.46125, 0.46625, 0.4525, 0.47625, 0.5025, 0.5325, 0.5225, 0.55375, 0.56375, 0.52875, 0.52375, 0.57, 0.5925, 0.57375, 0.59375, 0.55375, 0.595, 0.59875, 0.5775, 0.5825, 0.58875, 0.58875, 0.605, 0.605, 0.5725, 0.60875, 0.61125, 0.6175, 0.61375, 0.5925, 0.59875, 0.61625, 0.62, 0.62625, 0.62125])
        else:
            test_accuracy_values = accuracy_getter(path)[1]
            if label_values[index] == 'VGG':
                test_accuracy_values = output_scaler(test_accuracy_values)
        if label_values[index]=='Original ResNext' or label_values[index]=='Modified ResNext' or label_values[index]=='VGG':
            if label_values[index]=='VGG':
                test_accuracy_values = savgol_filter(test_accuracy_values,7,3)
            else:
                test_accuracy_values = savgol_filter(test_accuracy_values,11,3)
            plt.plot(x_vals,test_accuracy_values,label = label_values[index],color=colors[index])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim(0.9,0.96)
    plt.title('Test Accuracy Plots for Different Models')
    plt.legend(loc='lower right',fontsize=9)

    figure=plt.gcf()
    figure.set_size_inches(7.4, 4.4)
    plt.savefig('Different_ResNext.jpg',bbox_inches='tight',dpi = 700,figsize = (15,4.34))
    plt.clf()
