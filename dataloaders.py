import torch
import torchvision
import torchvision.transforms as transforms
import os

def dataloaders(main_file_directory,batch_size = 1):
    def return_loaders(transform_list,batch_size,main_file_directory='output'):
        train_data = torchvision.datasets.ImageFolder(root=os.path.join(main_file_directory,'train'), transform=transform_list[0])
        val_data = torchvision.datasets.ImageFolder(root=os.path.join(main_file_directory,'val'), transform=transform_list[1])
        test_data = torchvision.datasets.ImageFolder(root=os.path.join(main_file_directory,'test'),transform=transform_list[2])

        train_data_loader=torch.utils.data.DataLoader(train_data,
                                                      batch_size=batch_size,
                                                      shuffle=True,
                                                      num_workers=0,
                                                      drop_last=True)
        val_data_loader=torch.utils.data.DataLoader(val_data,
                                                      batch_size=batch_size,
                                                      shuffle=True,
                                                      num_workers=0,
                                                      drop_last=True)
        test_data_loader=torch.utils.data.DataLoader(test_data,
                                                      batch_size=batch_size,
                                                      shuffle=True,
                                                      num_workers=0,
                                                      drop_last=True)
        return train_data_loader,val_data_loader,test_data_loader

    def find_mean_std(loader):
        nimages = 0
        mean = 0.
        std = 0.
        for batch, _ in loader:
            # Rearrange batch to be the shape of [B, C, W * H]
            batch = batch.view(batch.size(0), batch.size(1), -1)
            # Update total number of images
            nimages += batch.size(0)
            # Compute mean and std here
            mean += batch.mean(2).sum(0)
            std += batch.std(2).sum(0)
        mean /= nimages
        std /= nimages
        print('mean',mean)
        print('std',std)
        return mean,std

    transform_list = [transforms.Resize((96,96)),transforms.ToTensor()]
    train_transforms, val_transforms, test_transforms = transform_list,transform_list,transform_list
    initial_transforms = transforms.Compose(transform_list)
    transform_list = [initial_transforms]*3

    train_data_loader, val_data_loader, test_data_loader = return_loaders(transform_list,batch_size = batch_size, main_file_directory = main_file_directory)

    train_mean,train_std = find_mean_std(train_data_loader)
    val_mean,val_std = find_mean_std(val_data_loader)
    test_mean,test_std = find_mean_std(test_data_loader)

    train_transforms = transforms.Compose(train_transforms + [transforms.Normalize(train_mean, train_std)])
    val_transforms = transforms.Compose(val_transforms + [transforms.Normalize(val_mean,val_std)])
    test_transforms = transforms.Compose(test_transforms + [transforms.Normalize(test_mean,test_std)])

    transform_list = [train_transforms, val_transforms, test_transforms]

    train_data_loader, val_data_loader, test_data_loader = return_loaders(transform_list,batch_size = batch_size, main_file_directory = main_file_directory)
    return train_data_loader, val_data_loader, test_data_loader

train_data_loader, val_data_loader, test_data_loader=dataloaders('output')
