import torch
def get_accuracy(predict_tensor,labels):
    _,predict = torch.max(predict_tensor,1)
    correct = (predict == labels).sum().item()
    return correct/len(predict)


def evaluate(model,val_loader,loss_function):
    total_acc = 0
    loss_out = 0
    with torch.no_grad():
        model.eval()
        for i,datapoint in enumerate(val_loader):
            inputs, label = datapoint
            label = label.long()

            inputs, label = inputs.cuda(), label.cuda()
            valid_predict = model(inputs.float())

            losstemp = loss_function(valid_predict,label)
            #print('evaluate loss:',losstemp)
            loss_out += losstemp.item()
            total_acc += get_accuracy(valid_predict, label)
        #print('total loss out:',loss_out)
    return float(total_acc)/(i+1), float(loss_out)/(i+1)
