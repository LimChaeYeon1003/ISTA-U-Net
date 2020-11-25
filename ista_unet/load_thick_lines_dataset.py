from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from os import listdir, path
import random
import numpy as np
from torch.utils.data.distributed import DistributedSampler

# Creat the noisy ellipsoid dataset
class thick_lines_rtc_dataset(Dataset):
    
    def __init__(self, mode, transform, train_percent = 0.8, dataset_dir = '/home/liu0003/Desktop/datasets/rtc/'):        
        super(thick_lines_rtc_dataset, self).__init__()
        self.dataset_str = 'thick_lines_rtc'
        self.transform = transform
        self.mode = mode

        thick_lines_source = np.squeeze(np.load(path.join(dataset_dir, 'explosion_thick_lines_30k_0p2_source_p.npy') ) )
        thick_lines_final = np.squeeze(np.load(path.join(dataset_dir, 'explosion_thick_lines_30k_0p2_final_p.npy') ) )

        self.train_size = int(train_percent * thick_lines_source.shape[0])
        self.test_size = thick_lines_source.shape[0] - self.train_size
        
        if self.mode == 'train':
            self.mode_thick_lines_source = thick_lines_source[:self.train_size]
            self.mode_thick_lines_final = thick_lines_final[:self.train_size]

        else:
            self.mode_thick_lines_source = thick_lines_source[self.test_size:]
            self.mode_thick_lines_final = thick_lines_final[self.test_size:]

    def __len__(self):
        if self.mode == 'train':
            num = self.train_size
        else:
            num = self.test_size
        return num
#     def __repr__(self):
#         return "thick_lines_rtc_dataset(mode={})". \
#             format(self.mode)

    def __getitem__(self, idx):
        
        source = self.transform(self.mode_thick_lines_source[idx]).type(torch.FloatTensor)
        final = self.transform(self.mode_thick_lines_final[idx]).type(torch.FloatTensor)

        return final, source


def get_dataloaders_thick_lines_rtc(batch_size=1, distributed_bool = False, **kwargs):

    batch_sizes = {'train': batch_size, 'test':1}

    train_transforms = transforms.Compose([
        transforms.ToTensor()])

    test_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    data_transforms = {'train': train_transforms,
                       'test': test_transforms}
    image_datasets = {'train': thick_lines_rtc_dataset(transform = data_transforms['train'], mode = 'train'),
                      'test': thick_lines_rtc_dataset(transform =  data_transforms['test'], mode = 'test')}
    
    if distributed_bool == True:
        dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_sizes[x], worker_init_fn = worker_init_fn, pin_memory=True, num_workers=0, sampler=DistributedSampler(image_datasets[x]) ) for x in ['train', 'test']}
    else:
        dataloaders = {x: DataLoader(image_datasets[x],  batch_size=batch_sizes[x], shuffle=(x == 'train'), worker_init_fn = worker_init_fn, pin_memory=True, num_workers=0 ) for x in ['train', 'test']}
    return dataloaders

def worker_init_fn(worker_id):
    random.seed(worker_id)
    np.random.seed(worker_id)
    torch.manual_seed(worker_id)
    torch.cuda.manual_seed_all(worker_id)
    torch.backends.cudnn.deterministic = True