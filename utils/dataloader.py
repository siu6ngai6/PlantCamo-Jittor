import os
from PIL import Image
from PIL import ImageOps
from jittor import transform
from jittor import dataset 
import jittor as jt
import numpy as np
import random


class CODataset(dataset.Dataset):
    """
    dataloader
    """
    def __init__(self, image_root, gt_root, batchsize, trainsize, shuffle, num_workers, augmentations):
        super().__init__()
        self.batch_size=batchsize
        self.trainsize = trainsize
        self.shuffle = shuffle
        self.num_workers = num_workers
        
        self.augmentations = augmentations
        print(self.augmentations)

        # self.images =self.read_files(image_root)
        # self.gts=self.read_files(gt_root)

        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        #
        self.images = sorted(self.images)
        #print('images path :',self.images)
        self.gts = sorted(self.gts)
        #print('gts path :',self.gts)
        self.filter_files()
        self.size = len(self.images)
        if self.augmentations == 'True':
            print('Using RandomRotation, RandomFlip')
            self.img_transform = transform.Compose([
                transform.RandomRotation(90, resample=False, expand=False, center=None, fill=None),
                transform.RandomVerticalFlip(p=0.5),
                transform.RandomHorizontalFlip(p=0.5),
                transform.Resize((self.trainsize, self.trainsize)),
                transform.ToTensor(),
                transform.ImageNormalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
            self.gt_transform = transform.Compose([
                transform.RandomRotation(90, resample=False, expand=False, center=None, fill=None),
                transform.RandomVerticalFlip(p=0.5),
                transform.RandomHorizontalFlip(p=0.5),
                transform.Resize((self.trainsize, self.trainsize)),
                transform.ToTensor()])
            
        else:
            print('no augmentation')
            self.img_transform = transform.Compose([
                transform.Resize((self.trainsize, self.trainsize)),
                transform.ToTensor(),
                transform.ImageNormalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
            
            self.gt_transform = transform.Compose([
                transform.Resize((self.trainsize, self.trainsize)),
                transform.ToTensor()])
            

    def __getitem__(self, index):
        
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        
        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        random.seed(seed) # apply this seed to img tranfsorms
        jt.set_global_seed(seed) # needed for torchvision 0.7
        if self.img_transform is not None:
            image = self.img_transform(image)
            
        random.seed(seed) # apply this seed to img tranfsorms
        jt.set_global_seed(seed) # needed for torchvision 0.7
        if self.gt_transform is not None:
            gt = self.gt_transform(gt)
        return image, gt

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            img = ImageOps.exif_transpose(img)
            gt = Image.open(gt_path)
            gt = ImageOps.exif_transpose(gt)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    # def read_files(self,root_file):
    #     sec_fold = os.listdir(root_file)
    #     sec_fold.sort()
    #     # print(sec_fold)
    #     image_path = []
    #     for sec in sec_fold:
    #         #print('root pth + sec',root_file + sec)
    #         #tempo_path = [root_file + sec + '/' + f for f in os.listdir(root_file + sec) if f.endswith('.jpg') or f.endswith('.png')]
    #         tempo_path = [root_file  + '/' + f for f in os.listdir(root_file ) if f.endswith('.jpg') or f.endswith('.png')]
    #         image_path = image_path + tempo_path
    #         #print(image_path, len(image_path))
    #     print(len(image_path))
    #     return image_path

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            img = ImageOps.exif_transpose(img)
            return img

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('L')
            img = ImageOps.exif_transpose(img)
            # return img.convert('1')
            return img

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, batchsize, trainsize, shuffle=True, num_workers=4, augmentation=False):
    data = CODataset(image_root, gt_root, batchsize, trainsize, shuffle, num_workers, augmentation)
    return data


class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transform.Compose([
            transform.Resize((self.testsize, self.testsize)),
            transform.ToTensor(),
            transform.ImageNormalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transform.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = jt.array(self.transform(image)).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        # self.index += 1
        # self.index = self.index % self.size
        # return image, gt, name
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            img = ImageOps.exif_transpose(img)
            return img

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('L')
            img = ImageOps.exif_transpose(img)
            return img

    def __len__(self):
        return self.size

class My_test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transform.Compose([
            transform.Resize((self.testsize, self.testsize)),
            transform.ToTensor(),
            transform.ImageNormalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transform.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = jt.array(self.transform(image)).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            img = ImageOps.exif_transpose(img)
            return img

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('L')
            img = ImageOps.exif_transpose(img)
            return img