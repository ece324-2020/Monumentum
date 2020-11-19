import torch
import torchvision
import torchvision.models as models
import torch.nn as nn
import ssl
import torchsummary

def return_model(model_tag='ResNet'):
    ssl._create_default_https_context = ssl._create_unverified_context

    res_mod = models.resnet34(pretrained=True)
    vgg_mod = models.vgg16(pretrained=True)
    print(res_mod.fc)
    num_ftrs = res_mod.fc.in_features
    res_mod.fc = nn.Sequential(nn.Linear(num_ftrs, 128),
                               nn.Linear(128,30))
    print(res_mod.fc)
    for name, child in res_mod.named_children():
        if name in ['fc']:
            print('{} has been unfrozen.'.format(name))
            for param in child.parameters():
                param.requires_grad = True
        else:
            for param in child.parameters():
                param.requires_grad = False

    models_dict = {'VGG':vgg_mod,'ResNet':res_mod}

    return models_dict[model_tag]

x = torch.randn(1,3,90,90)
res_mod = return_model()
y = res_mod(x)
print(y.shape)
