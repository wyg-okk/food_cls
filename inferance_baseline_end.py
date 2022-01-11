import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os, cv2, re, random, time
from model import resnet34
from torch.utils.data import Dataset
import config as cfg 
import torchvision
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


# 调用保存的最佳模型的准确率输出
def txt_dig(text):
    '''输入字符串，如果是数字则输出数字，如果不是则输出原本字符串'''
    return int(text) if text.isdigit() else text
def natural_keys(text):
    '''输入字符串，将数字与文字分隔开，将数字串转化为int'''
    return [ txt_dig(c) for c in re.split('(\d+)', text) ]
    
class LT_Dataset(Dataset):
    num_classes = cfg.num_classes

    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.targets = []
        self.transform = transform
        with open(root+txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.targets.append(int(line.split()[1]))
        
        cls_num_list_old = [np.sum(np.array(self.targets) == i) for i in range(self.num_classes)]
        
        # generate class_map: class index sort by num (descending)
        sorted_classes = np.argsort(-np.array(cls_num_list_old))
        self.class_map = [0 for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.class_map[sorted_classes[i]] = i
        
        self.targets = np.array(self.class_map)[self.targets].tolist()

        self.class_data = [[] for i in range(self.num_classes)]
        for i in range(len(self.targets)):
            j = self.targets[i]
            self.class_data[j].append(i)

        self.cls_num_list = [np.sum(np.array(self.targets)==i) for i in range(self.num_classes)]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        path = self.img_path[index]
        target = self.targets[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
        
class Food_LT(object):   
    def __init__(self, distributed, root="", batch_size=60, num_works=40): 
    
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
                ])
        test_txt = "/data/food/test.txt"		
        
        test_dataset = LT_Dataset(root, test_txt, transform=transform_test)	
        self.cls_num_list = test_dataset.cls_num_list
        self.dist_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset) if distributed else None
        #test_dataset = test_images 
        self.test = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_works, pin_memory=True)
        
def main():
    model=resnet34()
    ##TEST_DIR = './data/food/test/'#
    ##test_images=[TEST_DIR+i for i in os.listdir(TEST_DIR)]#
    ##test_images.sort(key=natural_keys)#####
   
    #预处理
    ##test=[]
    ##IMG_WIDTH = 128
    ##IMG_HEIGHT = 128
    ##for img in test_images:
    ##    test.append(cv2.resize(cv2.imread(img),(IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC))
    ##print('The shape of test data is {}'.format(np.array(test).shape))
    if cfg.gpu is not None:
        print('Use cuda !')
        torch.cuda.set_device(cfg.gpu)
        model = model.cuda(cfg.gpu)	
        
    print('Load dataset ...')    	
    dataset = Food_LT(False, root=cfg.root, batch_size=cfg.batch_size, num_works=4)
    test_loader = dataset.test	
    #####model.eval()		
	  #加载模型
    
    resume = './ckpt/model_best.pth.tar'
    checkpoint = torch.load(resume,map_location='cuda:0')#加载用训练数据训练好的模型
    model.load_state_dict(checkpoint['state_dict_model'])
    #用数据训练
    test_data_num = len(test_loader.dataset)
    end_steps = int(test_data_num / test_loader.batch_size)
    pred_class = np.array([],int)
    for i,images in enumerate(tqdm(test_loader)):
        if i > end_steps:
            break
        if torch.cuda.is_available():
            model.eval()
            if cfg.gpu is not None:
                images = images.cuda(cfg.gpu, non_blocking=True)
            output = model(images)           
            prob = torch.softmax(output, dim=1)
            confidence_part, pred_class_part = torch.max(prob, dim=1)
            pred_class = np.append(pred_class, pred_class_part.cpu().numpy())
    print('.......test end.......')
    print(pred_class) 
    path = "./data/food/test"
    id_name = os.listdir(path)
    dataframe = pd.DataFrame({'Id':id_name,'Expected':pred_class})
    dataframe.to_csv('./output/submission.csv',index=False,sep=',')
    
    print('test Finish !')

if __name__ == '__main__':
    main()
