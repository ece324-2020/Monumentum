from models.LeNet import LeNet
import torch
import global_vars as GLOBALS
import torch.nn as nn
from dataloaders import dataloaders
from torchsummary import summary
from utils import get_accuracy, evaluate
import time
import yaml
import matplotlib.pyplot as plt
#torch.set_num_threads(2)

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

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:',device)
    GLOBALS.CONFIG = initialize_hyper('config.yaml')
    LR = GLOBALS.CONFIG['baseline_LR']
    batch_size = GLOBALS.CONFIG['baseline_batch_size']
    epochs = GLOBALS.CONFIG['baseline_epochs']
    print('LR:{} | Batch Size:{} | Epochs:{}'.format(LR,batch_size,epochs))
    print('Preparing DataLoaders')
    train_loader, val_loader, test_loader = dataloaders('extracted_data_dir_split',batch_size=batch_size)
    print('DataLoaders Ready')
    LeNet_Baseline = LeNet()
    LeNet_Baseline = LeNet_Baseline.to(device)
    summary(LeNet_Baseline,(3,56,56))
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(LeNet_Baseline.parameters(),lr=LR)
    print('Starting Epochs')
    train_loss_store = []
    train_acc_store = []
    valid_loss_store = []
    valid_acc_store = []
    epoch_store = []
    for e in range(epochs):
        running_loss = 0
        running_acc = 0
        startepoch =time.time()
        for i,data in enumerate(train_loader):
            #print('Batch Number:',i)
            startepoch_Inner = time.time()
            #start = time.time()
            optimizer.zero_grad()
            img,labels = data
            #labels = labels.long()
            img = img.to(device)
            labels = labels.to(device)
            #print('time to ready data: {}'.format(time.time()-start))

            start=time.time()
            raw_predict = LeNet_Baseline(img)
            #print('time to get prediction data: {}'.format(time.time()-start))

            start=time.time()
            loss = loss_function(raw_predict,labels)
            loss.backward()
            optimizer.step()
            #print('Time to get loss and take backward step: {}'.format(time.time()-start))
            start=time.time()

            running_loss += loss.item()
            #print('predict:',raw_predict)
            #print('labels:',labels)

            running_acc += get_accuracy(raw_predict,labels)
            #print('Time to get acc and loss: {}'.format(time.time()-start))
            #end = time.time()
            #print('Batch Time Taken: {}'.format(end-startepoch_Inner))
            #print('Cumulative Epoch Time: {}'.format(end-startepoch))
            #print('Batch acc:',get_accuracy(raw_predict,labels))

        valid_acc,valid_loss = evaluate(LeNet_Baseline,val_loader,loss_function)

        print('Epoch {}/{} | Training Accuracy:{} | Training Loss:{} | Val Accuracy:{} | Val Loss:{} | Time Taken: {}'.format(e+1,
                                                            epochs,round(running_acc/(i+1),4),round(running_loss/(i+1),4),round(valid_acc,4),round(valid_loss,4),round(time.time()-startepoch,2)))
        train_loss_store.append(running_loss/(i+1))
        train_acc_store.append(running_acc/(i+1))
        valid_acc_store.append(valid_acc)
        valid_loss_store.append(valid_loss)
        epoch_store.append(e)
        if valid_acc >= 0.70:
            print('saving')
            torch.save(LeNet_Baseline.state_dict(),'baselineLeNet.pt')

    plt.plot(epoch_store,train_loss_store,label='Training Loss')
    plt.plot(epoch_store,valid_loss_store,label='Validation Loss')
    plt.title('Baseline LeNet: Loss VS Epochs (LR:{},batch_size:{},SGD)'.format(LR,batch_size))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='lower right')
    plt.savefig('BaselineLeNet_loss.png')
    plt.clf()
    plt.plot(epoch_store,train_acc_store,label='Training Accuracy')
    plt.plot(epoch_store,valid_acc_store,label='Validation Accuracy')
    plt.title('Baseline LeNet: Accuracy VS Epochs (LR:{},batch_size:{},SGD)'.format(LR,batch_size))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.savefig('BaselineLeNet_acc.png')
