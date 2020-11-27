import torch
import torchvision
import torchvision.models as models
import torch.nn as nn
import ssl
import torchsummary
import global_vars as GLOBALS
from dataloaders import dataloaders
from torchsummary import summary
from utils import get_accuracy, evaluate
import time
import yaml
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix
from shutil import copyfile

def create_confusion_plot(model,test_loader,classes,data_folder_path):
    classes = sorted([int(i) for i in classes])
    def confusion_matrix_generation(model,test_loader):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        y_true,y_pred=[],[]
        for i,data in enumerate(test_loader,0):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_true+=labels.tolist()
            y_pred+=predicted.tolist()
        confusion=confusion_matrix(y_true,y_pred)
        return confusion
    confusion = confusion_matrix_generation(model,test_loader)
    df_cm = pd.DataFrame(confusion, index = [i for i in classes],
                  columns = [i for i in classes])
    plt.figure(figsize = (11,7))
    sn.heatmap(df_cm, annot=True)
    plt.title('Confusion Plot for {} Model (Optim={}_LR={}_batchsize={}_momentum={})'.format(GLOBALS.CONFIG['model_name'],
                                                                                        GLOBALS.CONFIG['optim'],
                                                                                        GLOBALS.CONFIG['LR'],
                                                                                        GLOBALS.CONFIG['batch_size'],
                                                                                        GLOBALS.CONFIG['momentum']))
    plt.savefig(os.path.join(data_folder_path,'confusion_plot.jpg'),bbox_inches='tight',dpi=300)
    print('Done Confusion Plot')
    return True

def return_model(model_tag='ResNet'):
    ssl._create_default_https_context = ssl._create_unverified_context

    res_mod = models.resnet34(pretrained=True)
    vgg_mod = models.vgg16(pretrained=True)
    print(res_mod.fc, 'pre-change')
    num_ftrs = res_mod.fc.in_features
    res_mod.fc = nn.Sequential(nn.Linear(num_ftrs, 128),
                               nn.ReLU(),
                               nn.Linear(128,26))
    print(res_mod.fc)
    vgg_mod.classifier = nn.Sequential(nn.Linear(25088,4096,bias=True),
                               nn.ReLU(),
                               nn.Dropout(p=0.5),
                               nn.Linear(4096,4096,bias=True),
                               nn.ReLU(),
                               nn.Dropout(p=0.5),
                               nn.Linear(4096,27,bias=True)
                               )
    for name, child in vgg_mod.named_children():
        if name in ['classifier']:
            print('{} has been unfrozen.'.format(name))
            print(child.parameters())
            for param in child.parameters():
                param.requires_grad = True
        else:
            for param in child.parameters():
                param.requires_grad = False
    for name, child in res_mod.named_children():
        if name in ['fc']:
            print('{} has been unfrozen.'.format(name))
            for param in child.parameters():
                param.requires_grad = True
        else:
            for param in child.parameters():
                param.requires_grad = False

    models_dict = {'VGG16':vgg_mod,'ResNet34':res_mod}

    return models_dict[model_tag]


#torch.set_num_threads(2)
'''
x = torch.randn(1,3,90,90)
res_mod = return_model()
y = res_mod(x)
print(y.shape)
'''
def initialize_hyper(path_to_config):
    '''
    Reads config.yaml to set hyperparameters
    '''
    with open(path_to_config, 'r') as stream:
        try:
            GLOBALS.CONFIG = yaml.safe_load(stream)
            return GLOBALS.CONFIG
        except yaml.YAMLError as exc:
            print(exc)
            return None

def initialize_training(input_model='ResNet',optimizer_tag='SGD',momentum_tag = 0):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:',device)
    LR = GLOBALS.CONFIG['LR']
    batch_size = GLOBALS.CONFIG['batch_size']
    epochs = GLOBALS.CONFIG['epochs']
    print('LR:{} | Batch Size:{} | Epochs:{} | Momentum:{} | Optim:{}'.format(LR,batch_size,epochs,momentum_tag,optimizer_tag))
    print('Preparing DataLoaders')
    train_loader, val_loader, test_loader = dataloaders('dataset_delf_filtered_augmented_split',batch_size=batch_size)
    classes = os.listdir('dataset_delf_filtered_augmented_split'+os.sep+'train')
    try:
        classes.remove('.DS_Store')
    except:
        pass
    loaders = {'train':train_loader, 'val':val_loader, 'test':test_loader}
    print('DataLoaders Ready')
    mod = return_model(model_tag=input_model)
    mod = mod.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizers = {'SGD':torch.optim.SGD,'Adam':torch.optim.Adam,'RMSProp':torch.optim.RMSprop}
    if optimizer_tag!='Adam':
        optimizer = optimizers[optimizer_tag](mod.parameters(),momentum=momentum_tag,lr=LR)
    else:
        optimizer = optimizers[optimizer_tag](mod.parameters(),lr=LR)
    return mod, loss_function, optimizer, loaders, classes, device

if __name__ == '__main__':
    performance_statistics = {}
    base = os.path.dirname(os.path.realpath(__file__))
    output_path = os.path.join(base,'train_data_files')
    try:
        os.mkdir(output_path)
    except OSError as error:
        pass

    GLOBALS.CONFIG = initialize_hyper('config.yaml')

    sub_folder_path = 'Model={}_Optim={}_LR={}_batchsize={}_epochs={}_momentum={}'.format(GLOBALS.CONFIG['model_name'],GLOBALS.CONFIG['optim'],GLOBALS.CONFIG['LR'],GLOBALS.CONFIG['batch_size'],GLOBALS.CONFIG['epochs'],GLOBALS.CONFIG['momentum'])

    try:
        os.mkdir(os.path.join(output_path,sub_folder_path))
    except:
        pass

    data_folder_path = os.path.join(output_path,sub_folder_path)

    model, loss_function, optimizer, loaders, classes, device = initialize_training(input_model=GLOBALS.CONFIG['model_name'],
                    optimizer_tag=GLOBALS.CONFIG['optim'],momentum_tag = GLOBALS.CONFIG['momentum'])


    print('Starting Epochs')
    train_loss_store = []
    train_acc_store = []
    valid_loss_store = []
    valid_acc_store = []
    test_loss_store = []
    test_acc_store = []
    epochs = GLOBALS.CONFIG['epochs']
    epoch_store = []
    for e in range(epochs):
        running_loss = 0
        running_acc = 0
        startepoch =time.time()
        for i,data in enumerate(loaders['train']):
            startepoch_Inner = time.time()
            optimizer.zero_grad()
            img,labels = data
            img = img.to(device)
            labels = labels.to(device)

            raw_predict = model(img)

            loss = loss_function(raw_predict,labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            running_acc += get_accuracy(raw_predict,labels)

        train_accuracy = round(running_acc/(i+1),4)
        train_loss = round(running_loss/(i+1),4)
        valid_acc,valid_loss = evaluate(model,loaders['val'],loss_function)
        test_acc,test_loss = evaluate(model,loaders['test'],loss_function)

        performance_statistics['acc_epoch_' + str(e)] = train_accuracy
        performance_statistics['train_loss_epoch_' + str(e)] = train_loss
        performance_statistics['valid_accuracy_epoch_' + str(e)] = round(valid_acc,4)
        performance_statistics['valid_loss_epoch_' + str(e)] = round(valid_loss,4)
        performance_statistics['test_accuracy_epoch_' + str(e)] = round(test_acc,4)
        performance_statistics['test_loss_epoch_' + str(e)] = round(test_loss,4)



        print('Epoch {}/{} | Training Accuracy:{} | Training Loss:{} | Val Accuracy:{} | Val Loss:{} | Time Taken: {}'.format(e+1,
                                                            epochs,train_accuracy,train_loss,round(valid_acc,4),round(valid_loss,4),round(time.time()-startepoch,2)))
        train_loss_store.append(running_loss/(i+1))
        train_acc_store.append(running_acc/(i+1))
        valid_acc_store.append(valid_acc)
        valid_loss_store.append(valid_loss)
        test_acc_store.append(test_acc)
        test_loss_store.append(test_loss)
        epoch_store.append(e)


    #torch.save(model.state_dict(),os.path.join(data_folder_path,'{}.pt'.format(GLOBALS.CONFIG['model_name'])))
    copyfile('config.yaml',os.path.join(data_folder_path,'config_we_used.yaml'))

    df = pd.DataFrame([performance_statistics])
    xlsx_name = 'LR={}_batchsize={}_epochs={}.xlsx'.format(GLOBALS.CONFIG['LR'],GLOBALS.CONFIG['batch_size'],GLOBALS.CONFIG['epochs'])
    writer = pd.ExcelWriter(os.path.join(data_folder_path,xlsx_name), engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()
    print('Test Acc: {} | Test Loss: {}'.format(test_acc,test_loss))

    plt.plot(epoch_store,train_loss_store,label='Training Loss')
    plt.plot(epoch_store,valid_loss_store,label='Validation Loss')
    plt.plot(epoch_store,test_loss_store,label='Test Loss')
    plt.title('{}: Loss VS Epochs (LR:{},batch_size:{},optim:{})'.format(GLOBALS.CONFIG['model_name'],
                                                                        GLOBALS.CONFIG['LR'],
                                                                        GLOBALS.CONFIG['batch_size'],
                                                                        GLOBALS.CONFIG['optim']))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='lower left')
    plt.savefig(os.path.join(data_folder_path,'{}_{}_{}_{}_loss.png'.format(GLOBALS.CONFIG['model_name'],
                                    GLOBALS.CONFIG['LR'],
                                    GLOBALS.CONFIG['batch_size'],
                                    GLOBALS.CONFIG['optim'])),bbox_inches='tight',dpi=300)
    plt.clf()
    plt.plot(epoch_store,train_acc_store,label='Training Accuracy')
    plt.plot(epoch_store,valid_acc_store,label='Validation Accuracy')
    plt.plot(epoch_store,test_acc_store,label='Test Accuracy')
    plt.title('{}: Acc VS Epochs (LR:{},batch_size:{},optim:{})'.format(GLOBALS.CONFIG['model_name'],
                                                                        GLOBALS.CONFIG['LR'],
                                                                        GLOBALS.CONFIG['batch_size'],
                                                                        GLOBALS.CONFIG['optim']))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(data_folder_path,'{}_{}_{}_{}_acc.png'.format(GLOBALS.CONFIG['model_name'],
                                    GLOBALS.CONFIG['LR'],
                                    GLOBALS.CONFIG['batch_size'],
                                    GLOBALS.CONFIG['optim'])),bbox_inches='tight',dpi=300)
    create_confusion_plot(model,loaders['test'],classes,data_folder_path)
