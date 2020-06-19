from torch.utils import data
import torch
import numpy as np
from scipy.io import loadmat, savemat



class Patches(data.TensorDataset):
    """Build dataset for MR images segmentation
       Args:
        mrsi: metabolite spectrum of brain that has size [x,y,z,t] = [78,110,~20,142]
        data1: flair image that has original size of [xd,yd,zd] = [256,256,80] and resized size of [3x, 3y, 3z]
        ground truth: manual label at data1's spatial resolution
    """

    def __init__(self, data_dir, subjects, patch_size):
        """Initialize dataset"""

        self.data_dir = data_dir
        self.subjects = subjects
        self.patch_size = patch_size
        diction, subnum, dataset = [], 0, []
        # produce patches
        for n in subjects:
            # loadmat returns a dictionary, the value is an np.array
            mrsi = loadmat(data_dir + 'stroke' + str(n) + '_spice.mat')
            mrsi = mrsi['xf_data']
            data1 = loadmat(data_dir + 'stroke' + str(n) + '_anatRef.mat')
            data1 = data1['anatRef']
            data1 = data1[:,:,:,np.newaxis]
            gt = loadmat(data_dir + 'stroke' + str(n) + '_mask.mat')
            gt = gt['core_mask']
            mask = loadmat(data_dir + 'stroke' + str(n) + '_dilate.mat')
            mask = mask['wholeMask']
            combined = np.concatenate((data1, mrsi),3)
            x,y,z,_ = combined.shape
            combined = np.transpose(combined, axes=[3,0,1,2])
            radius = int(patch_size / 2)
            for i in range(radius, x-radius):
                for j in range(radius, y-radius):
                    for k in range(radius, z-radius):
                        if mask[i,j,k]:
                            diction.append([i,j,k,subnum])
            dataset.append(combined)

        self.dataset = dataset
        self.gt = gt
        savemat('gt.mat', {'gt':self.gt})
        self.diction = diction
        self.num_patches = len(diction)
        print('total data size:' + str(len(diction)))


    def __getitem__(self, index):
        """Return ith patch of the dataset"""
        
        i,j,k,subnum = self.diction[index]
        combined = self.dataset
        gt = self.gt
        radius = int(self.patch_size / 2)

        train_data = abs(combined[subnum][:,i-radius:i+radius, j-radius:j+radius, k-radius:k+radius])
        train_data = torch.tensor(train_data, dtype=torch.long)

        target = gt[i-radius:i+radius, j-radius:j+radius, k-radius:k+radius]
        target = torch.tensor(target, dtype=torch.float)

        return [train_data, target, i, j, k]

    def __len__(self):
        """Return the length of dataset"""

        return self.num_patches

def get_loader(data_dir, subjects, patch_size, mode='train', batch_size=16, num_workers=0, train_percentage=0.8):

    """Build and return a data loader."""

    if mode == 'train_test':
        dataset = Patches(data_dir, subjects, patch_size)
        train_size = int(dataset.num_patches * train_percentage)
        test_size = dataset.num_patches - train_size

        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        train_data_loader = data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=num_workers,
                                            drop_last=True)

        test_data_loader = data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=num_workers,
                                            drop_last=True)

        return (train_data_loader, test_data_loader)

    else:
        dataset = Patches(data_dir, subjects, patch_size)
        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=num_workers,
                                      drop_last=True)
        return data_loader


