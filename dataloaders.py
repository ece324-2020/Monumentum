import torch
import torchvision
import torchvision.transforms as transforms
import os

def dataloaders(main_file_directory,batch_size = 1):
    def return_loaders(transform_list,batch_size,main_file_directory='output'):
        print('Batch Size used in return loader:',batch_size)
        train_data = torchvision.datasets.ImageFolder(root=os.path.join(main_file_directory,'train'), transform=transform_list[0])
        val_data = torchvision.datasets.ImageFolder(root=os.path.join(main_file_directory,'val'), transform=transform_list[1])
        test_data = torchvision.datasets.ImageFolder(root=os.path.join(main_file_directory,'test'),transform=transform_list[2])

        train_data_loader=torch.utils.data.DataLoader(train_data,
                                                      batch_size=batch_size,
                                                      shuffle=True,
                                                      num_workers=4,
                                                      drop_last=True,
                                                      pin_memory=True)
        val_data_loader=torch.utils.data.DataLoader(val_data,
                                                      batch_size=batch_size,
                                                      shuffle=True,
                                                      num_workers=4,
                                                      drop_last=True,
                                                      pin_memory=True)
        test_data_loader=torch.utils.data.DataLoader(test_data,
                                                      batch_size=batch_size,
                                                      shuffle=True,
                                                      num_workers=4,
                                                      drop_last=True,
                                                      pin_memory=True)
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
        #print('mean',mean)
        #print('std',std)
        return mean,std
    torch.manual_seed(1234)
    training_transform_list = [transforms.Resize((56,56)),
                              transforms.RandomHorizontalFlip(p=0.4),
                              #transforms.GaussianBlur(3,sigma=1/3),
                              transforms.ToTensor()]
    val_test_transform_list = [transforms.Resize((56,56)),transforms.ToTensor()]

    train_transforms, val_transforms, test_transforms = training_transform_list,val_test_transform_list,val_test_transform_list
    '''
    initial_train_transforms = transforms.Compose(training_transform_list)
    initial_val_test_transforms = transforms.Compose(val_test_transform_list)

    transform_list = [initial_train_transforms, initial_val_test_transforms, initial_val_test_transforms]
    # Call to get data loaders so we can calculate their mean and std
    train_data_loader, val_data_loader, test_data_loader = return_loaders(transform_list,batch_size = batch_size, main_file_directory = main_file_directory)
    print('Finding  stats')
    train_mean,train_std = find_mean_std(train_data_loader)
    print('training mean:{} and std:{}'.format(train_mean,train_std))
    val_mean,val_std = find_mean_std(val_data_loader)
    print('Val mean:{} and std:{}'.format(val_mean,val_std))
    test_mean,test_std = find_mean_std(test_data_loader)
    print('Test mean:{} and std:{}'.format(test_mean,test_std))
    '''
    # Normalize with calculated mean and std (approximate calculated on subset of data)
    train_transforms = transforms.Compose(train_transforms + [transforms.Normalize([0.475,0.48,0.465],[0.215,0.22,0.245])])
    val_transforms = transforms.Compose(val_transforms + [transforms.Normalize([0.475,0.48,0.465],[0.215,0.22,0.245])])
    test_transforms = transforms.Compose(test_transforms + [transforms.Normalize([0.475,0.48,0.465],[0.215,0.22,0.245])])

    transform_list = [train_transforms, val_transforms, test_transforms]

    train_data_loader, val_data_loader, test_data_loader = return_loaders(transform_list,batch_size = batch_size, main_file_directory = main_file_directory)
    return train_data_loader, val_data_loader, test_data_loader

if __name__ == '__main__':
    train_data_loader, val_data_loader, test_data_loader=dataloaders('data_main_split',batch_size = 128)
