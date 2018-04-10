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

class Mydataset(torch.utils.data.Dataset):
	def My_pkl_loader(self, path):
		f = open(path, 'rb')
		return pickle.load(f)

	def __init__(self, phase='train'):#, batch_size=256, workers=6, worker_buf=128):
		super(Mydataset, self).__init__()
		self.phase = phase
		#self.batch_size = batch_size
		#self.workers = workers if phase == 'train' else 1
		#self.worker_buf = worker_buf
		
		self.mean = [.485, .456, .406]
		self.std = [.229, .224, .225]
		self.normalize = transforms.Normalize(mean=self.mean, std=self.std)

		self.root_dir = '/n/jbr/CVDLdata/hw1/'
		self.data_dir = self.root_dir + 'data/'
		
		# Train configures
		self.train_img_dir = self.data_dir + 'train/'
		self.train_anns = self.My_pkl_loader(self.data_dir + 'traininfo.pkl')
		self.train_size = len(self.train_anns)
		self.train_transform = transforms.Compose([
			transforms.RandomResizedCrop(224),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			self.normalize,
		])
		
		# Test configures
		self.test_img_dir = self.data_dir + 'val/'
		self.test_anns = self.My_pkl_loader(self.data_dir + 'valinfo.pkl')
		self.test_size = len(self.test_anns)
		self.test_transform = transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			self.normalize,
		])
		
		# Starting processing
		'''
		batch_size = self.batch_size if batch_size is None else batch_size

		processors = []
		result_queue = Queue(self.worker_buf)
		for pid in range(self.workers):
			if self.phase == 'train':
				p = Process(target=worker, args=(
					pid, self.phase, batch_size, self.data_dir, 
					self.train_anns.copy(), self.train_transform,
					result_queue
				))
			else:
				p = Process(target=worker, args=(
					pid, self.phase, 1, self.data_dir,
					self.test_anns.copy(), self.test_transform,
					result_queue
				))	  
			p.start()
			processors.append(p)
		'''


	def __getitem__(self, ind):
		if self.phase == 'train':
			info = self.train_anns[ind]
			transform = self.train_transform
		else:
			info = self.test_anns[ind]
			transform = self.test_transform
		
		img = None
		cnt = 0
		while img is None:
			cnt += 1
			img = cv2.imread(self.data_dir+info[0])
			if cnt >= 10000:
				print('fuck'+self.data_dir+info[0])
		img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
		inp = torch.FloatTensor(transform(img))
		target = long(info[1])

		return inp, target


	def __len__(self):
		if self.phase == 'train':
			return self.train_size
		else:
			return self.test_size

	'''
	def next_batch(self):
		while True:
			data, target = result_queue.get()
		
		for p in processors:
			p.join()
	'''

