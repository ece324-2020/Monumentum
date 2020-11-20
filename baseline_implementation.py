from models.LeNet import LeNet
import torch
import torch.nn as nn
import torchvision.models as models
from dataloaders import dataloaders
from torchsummary import summary
from utils import get_accuracy, evaluate
import time
#torch.set_num_threads(2)
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:',device)
    LR = 0.2
    batch_size = 64
    epochs = 200
    print('Preparing DataLoaders')
    train_loader, val_loader, test_loader = dataloaders('extracted_data_dir_split_subset',batch_size=batch_size)
    print('DataLoaders Ready')
    LeNet_Baseline = LeNet()
    LeNet_Baseline = LeNet_Baseline.to(device)
    summary(LeNet_Baseline,(3,56,56))
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(LeNet_Baseline.parameters(),lr=LR)
    print('Starting Epochs')
    #freeze_support()
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
        print('------------------------------------')
        print('Epoch {}/{} | Training Accuracy:{} | Training Loss:{} | Val Accuracy:{} | Val Loss:{} | Time Taken: {}'.format(e+1,
                                                            epochs,round(running_acc/(i+1),4),round(running_loss/(i+1),4),round(valid_acc,4),round(valid_loss,4),round(time.time()-startepoch,2)))
        print('------------------------------------')
