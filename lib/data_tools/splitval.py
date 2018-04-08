# Title: 	train & val dataset (from original train) generator
# Author:	Jiang Borui
# Date:		2018/04/06
import os
import pickle
import random

def SavePickle(A, path):
	f = open(path, 'wb')
	pickle.dump(A, f)

lst = []
with open('train.info', 'r') as f:
	lines = f.readlines()
	for line in lines:
		temp = line.split(' ')
		temp[1] = int(temp[1][:-1])
		lst.append(temp)
random.seed(5)
random.shuffle(lst)
valset = lst[:2000]
trainset = lst[2000:]

for i in range(len(valset)):
	info = valset[i]
	filename = info[0].split('/')[-1]
	#os.system('cp '+info[0]+' split/val/'+filename)
	#print('cp '+info[0]+' split/val/'+filename+', translate '+valset[i][0]+' to val/')
	valset[i][0] = 'val/'+filename
for i in range(len(trainset)):
	info = trainset[i]
	filename = info[0].split('/')[-1]
	#os.system('cp '+info[0]+' split/train/'+filename)
	#print('cp '+info[0]+' split/train/'+filename+', translate '+trainset[i][0]+' to train/')
	trainset[i][0] = 'train/'+filename
print('Saving')
SavePickle(valset, 'split/valinfo.pkl')
SavePickle(trainset, 'split/traininfo.pkl')
print('Done')
