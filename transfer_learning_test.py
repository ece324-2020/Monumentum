import torch
import torchvision
import torchvision.models as models
import torch.nn as nn
import ssl

def return_model(model_tag='ResNet'):
    ssl._create_default_https_context = ssl._create_unverified_context

    res_mod = models.resnet34(pretrained=True)
    vgg_mod = models.vgg16(pretrained=True)
    resnext_mod = models.resnext101_32x8d(pretrained=True)
    print(res_mod.fc, 'pre-change')
    num_ftrs = res_mod.fc.in_features
    res_mod.fc = nn.Sequential(nn.Linear(num_ftrs, 128),
                               nn.Linear(128,26))
    vgg_mod.classifier = nn.Sequential(nn.Linear(25088,4096,bias=True),
                               nn.ReLU(),
                               nn.Dropout(p=0.5),
                               nn.Linear(4096,4096,bias=True),
                               nn.ReLU(),
                               nn.Dropout(p=0.5),
                               nn.Linear(4096,27,bias=True)
                               )
    resnext_mod.fc = nn.Sequential(nn.Linear(2048,26,bias=True))
    exit()
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
    for name, child in res_mod.named_children():
        if name in ['fc']:
            print('{} has been unfrozen.'.format(name))
            for param in child.parameters():
                param.requires_grad = True
        else:
            for param in child.parameters():
                param.requires_grad = False
    models_dict = {'VGG16':vgg_mod,'ResNet34':res_mod,'ResNext101':resnext_mod}}

    return models_dict[model_tag]

return_model()
