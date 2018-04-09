# Title:	train & test provider for CVDL hw1
# Author:	Jiang Borui
# Date:		2018/04/08
# this code is multiprocessing iterative data provider for train.py

import os
import cv2
import time
import random
import pickle
import numpy as np
from PIL import Image
from multiprocessing import Process, Queue

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable

def worker(pid, batch_size, data_dir, img_anns, transformer, result_queue):
    data = torch.FloatTensor(batch_size, 3, 224, 224).zero_()
    target = torch.LongTensor(batch_size).zero_()
   
    print('\033[0;32;40m[Provider][Processor %d]: Starting...\033[0m' % pid)
    random.seed(pid)
    random.shuffle(img_anns)
    img_batch_ptr = 0
    img_processor_ptr = 0
    img_anns_size = len(img_anns)
    while True:
        bid = 0
        while bid < batch_size:
            info = img_anns[img_batch_ptr]
            
            img_processor_ptr += 1
            if img_processor_ptr % 10000 == 0:
                print('\033[0;32;40m[Provider][Processor %d]: Already put %d images.\033[0m' % (
                    pid, img_processor_ptr))

            img_batch_ptr += 1
            if img_batch_ptr == img_anns_size:
                random.shuffle(img_anns)
                img_batch_ptr = 0

            img = cv2.imread(data_dir+info[0])
            if img is None:
                continue
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            inp = transformer(img)

            data[bid] = inp
            target[bid] = long(info[1])
            bid += 1
        result_queue.put_nowait((data, target))
    return

class Provider:
    def My_pkl_loader(self, path):
        f = open(path, 'rb')
        return pickle.load(f)

    def __init__(self, phase='train', batch_size=256, workers=6, worker_buf=32*256):
        self.phase = phase
        self.batch_size = batch_size
        self.workers = workers if phase == 'train' else 1
        self.worker_buf = worker_buf
        
        self.mean = [.485, .456, .406]
        self.std = [.229, .224, .225]
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)

        self.root_dir = '/n/jbr/CVDLdata/hw1/'
        self.data_dir = self.root_dir + 'data/split/'
        
        # Train configures
        self.train_img_dir = self.data_dir + 'train/'
        self.train_anns = self.My_pkl_loader(self.data_dir + 'traininfo.pkl')
        self.train_size = len(self.train_anns)
        self.train_transform = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize,
        ])
        
        # Test configures
        self.test_img_dir = self.data_dir + 'val/'
        self.test_anns = self.My_pkl_loader(self.data_dir + 'valinfo.pkl')
        self.test_size = len(self.test_anns)
        self.test_transform = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            self.normalize,
        ])

    def next_batch(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size

        processors = []
        result_queue = Queue(self.worker_buf)
        for pid in range(self.workers):
            if self.phase == 'train':
                p = Process(target=worker, args=(
                    pid, batch_size, self.data_dir, 
                    self.train_anns, self.train_transform,
                    result_queue
                ))
            else:
                p = Process(target=worker, args=(
                    pid, 1, self.data_dir,
                    self.test_anns, self.test_transform,
                    result_queue
                ))    
            p.start()
            processors.append(p)

        while True:
            data, target = result_queue.get()
            yield data, target
        
        for p in processors:
            p.join()
                

