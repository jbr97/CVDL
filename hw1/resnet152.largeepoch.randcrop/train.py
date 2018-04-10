# Title:	Resnet example
# Author:	Borui Jiang
# Date:		2018/04/08
# this code is modified from the pytorch example code (by Bolei Zhou): https://github.com/CSAILVision/places365/blob/master/train_placesCNN.py

import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import wideresnet
from dataset import Mydataset
import pdb
model_names = sorted(name for name in models.__dict__
	if name.islower() and not name.startswith("__")
	and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch ResNet50 (for AI challenge) Training')
parser.add_argument('--data', metavar='DIR', type=str, default='/n/jbr/CVDLdata/hw1/data/split',
					help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet152', type=str,
					help='model architecture: ' +
						' | '.join(model_names) +
						' (default: resnet18)')
parser.add_argument('-j', '--workers', default=6, type=int, metavar='N',
					help='number of data loading workers (default: 6)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
					help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
					help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
					metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
					metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
					help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
					metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
					metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
					help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
					help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_false',
					help='use pre-trained model')
parser.add_argument('--num_classes',default=80, type=int, help='num of class in the model')

best_prec1 = 0


def main():
	global args, best_prec1

	print('parsing args...')
	args = parser.parse_args()
	
	# create model
	print("=> creating model '{}'".format(args.arch))
	if args.arch.lower().startswith('wideresnet'):
		# a customized resnet model with last feature map size as 14x14 for better class activation mapping
		model  = wideresnet.resnet50(num_classes=args.num_classes)
	else:
		model = models.__dict__[args.arch](num_classes=args.num_classes)

	if args.arch.lower().startswith('alexnet') or args.arch.lower().startswith('vgg'):
		model.features = torch.nn.DataParallel(model.features)
		model.cuda()
	else:
		model = torch.nn.DataParallel(model).cuda()
		model = model.cuda()
	print(model)

	# optionally resume from a checkpoint
	if args.resume:
		if os.path.isfile(args.resume):
			print("=> loading checkpoint '{}'".format(args.resume))
			checkpoint = torch.load(args.resume)
			args.start_epoch = checkpoint['epoch']
			best_prec1 = checkpoint['best_prec1']
			model.load_state_dict(checkpoint['state_dict'])
			print("=> loaded checkpoint '{}' (epoch {})"
						  .format(args.resume, checkpoint['epoch']))
		else:
			print("=> no checkpoint found at '{}'".format(args.resume))

	cudnn.benchmark = True
	# Data loading code
	#train_loader = Provider(phase = 'train', batch_size=args.batch_size, workers=args.workers)
	#val_loader = Provider(phase = 'test', batch_size=args.batch_size)
	train_loader = torch.utils.data.DataLoader(
		Mydataset(phase='train'), 
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=args.workers,
		pin_memory=True)
	
	val_loader = torch.utils.data.DataLoader(
		Mydataset(phase='test'), 
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=args.workers,
		pin_memory=True)
	

	# define loss function (criterion) and pptimizer
	criterion = nn.CrossEntropyLoss().cuda()

	optimizer = torch.optim.SGD(model.parameters(), args.lr,
		momentum=args.momentum,
		weight_decay=args.weight_decay)

	if args.evaluate:
		validate(val_loader, model, criterion)
		return

	for epoch in range(args.start_epoch, args.epochs):
		adjust_learning_rate(optimizer, epoch)

		# train for one epoch
		train(train_loader, model, criterion, optimizer, epoch)

		# evaluate on validation set
		prec1 = validate(val_loader, model, criterion)

		# remember best prec@1 and save checkpoint
		is_best = prec1 > best_prec1
		best_prec1 = max(prec1, best_prec1)
		save_checkpoint({
			'epoch': epoch,
			'arch': args.arch,
			'state_dict': model.state_dict(),
			'best_prec1': best_prec1,
		}, is_best, './snapshot/'+args.arch.lower()+'_'+str(epoch))


def train(train_loader, model, criterion, optimizer, epoch):
	global args

	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	top3 = AverageMeter()

	# switch to train mode
	model.train()

	end = time.time()
	for i, (inp, target) in enumerate(train_loader):
		# measure data loading time
		data_time.update(time.time() - end)

		#inp = inp.cuda(async=True)
		target = target.cuda(async=True)
		inp_var = torch.autograd.Variable(inp)
		target_var = torch.autograd.Variable(target)
		# compute output
		output = model(inp_var)
		loss = criterion(output, target_var)

		# measure accuracy and record loss
		prec1, prec3 = accuracy(output.data, target, topk=(1, 5))
		losses.update(loss.data[0], inp.size(0))
		top1.update(prec1[0], inp.size(0))
		top3.update(prec3[0], inp.size(0))

		# compute gradient and do SGD step
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % args.print_freq == 0:
			print('Epoch: [{0}][{1}/{2}]\t'
				'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
				'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
				'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
				epoch, i, len(train_loader), batch_time=batch_time,
				data_time=data_time, loss=losses, top1=top1, top3=top3))


def validate(val_loader, model, criterion):
	batch_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	top3 = AverageMeter()

	# switch to evaluate mode
	model.eval()

	end = time.time()
	for i, (inp, target) in enumerate(val_loader):
		#inp = inp.cuda(async=True)
		target = target.cuda(async=True)
		inp_var = torch.autograd.Variable(inp, volatile=True)
		target_var = torch.autograd.Variable(target, volatile=True)

		# compute output
		output = model(inp_var)
		loss = criterion(output, target_var)

		# measure accuracy and record loss
		prec1, prec3 = accuracy(output.data, target, topk=(1, 3))
		losses.update(loss.data[0], inp.size(0))
		top1.update(prec1[0], inp.size(0))
		top3.update(prec3[0], inp.size(0))

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % args.print_freq == 0:
			print('Test: [{0}/{1}]\t'
				'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
				'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
				i, len(val_loader), batch_time=batch_time, loss=losses,
				top1=top1, top3=top3))

	print(' * Prec@1 {top1.avg:.3f} Prec@5 {top3.avg:.3f}'
		.format(top1=top1, top3=top3))

	return top3.avg


def save_checkpoint(state, is_best, fileprefix='checkpoint'):
	filename = fileprefix + '.pth.tar'
	torch.save(state, filename)
	if is_best:
		shutil.copyfile(filename, 'snapshot/snapshot_best.pth.tar')


class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
	"""Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
	lr = args.lr * (0.1 ** (epoch // 30))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res


if __name__ == '__main__':
	main()
