#coding = utf-8

import os
from PIL import Image
import json
import time
from skimage.transform import resize
import numpy as np
from glob import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils import data
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
from models import stnet
import datetime
import visdom

BATCH_SIZE = 10


vis = visdom.Visdom(env='StNet')
x = 0
y = 0
val_loss = vis.line(
    X=np.array([x]),
    Y=np.array([y]),
    opts=dict(title='The loss of the val'))

train_loss = vis.line(
    X=np.array([x]),
    Y=np.array([y]),
    opts=dict(title='The loss of the training'))

train_acc = vis.line(
    X=np.array([x]),
    Y=np.array([y]),
    opts=dict(title='The accuracy of the training'))

val_acc = vis.line(
    X=np.array([x]),
    Y=np.array([y]),
    opts=dict(title='The accuracy of the val'))

# read the frames and label of each clips, then stack them and toTensor
class videoDataSet(Dataset):
    def __init__(self,video_info_list,transform=None):
        self.video_info_list = video_info_list
        self.transform = transform

    def __len__(self):
        return len(self.video_info_list)

    def __getitem__(self, idx):
        video_info = self.video_info_list[idx]
        video_name = video_info['video_name']
        label = video_info['label']
        video_file = video_name[0:-4]
        clip = []
        for num in range(35):
            pic_num = str(num+1) + '.jpg'
            clip.append(os.path.join('data', 'ucf', video_file, pic_num))
            #clip = sorted (glob (os.path.join ('data', 'ucf', video_file, '*.jpg')))
        video_sequence = np.array([self.transform(Image.open(frame)).numpy() for frame in clip])
        video_sequence = video_sequence.transpose(1, 0, 2, 3)   # C, F, H, W
        video_sequence = torch.from_numpy(video_sequence)
        return video_sequence, label

def train_model(model, criterion, optimizer, scheduler,num_epochs=1):
    since = time.time()
    best_acc = 0

    for epoch in range(num_epochs):
        epoch += 1
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # each epoch has train and val
        for mode in ['train', 'val']:
            if mode == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0
            running_correct = 0

            for item, datas in enumerate(data_loaders[mode]):

                videos = datas[0]
                labels = datas[1]

                # use gpu
                if use_gpu:
                    videos = Variable(videos.cuda())
                    labels = Variable(labels.cuda())
                else:
                    videos = Variable(videos)
                    labels = Variable(labels)

                optimizer.zero_grad()
                if mode == 'train':
                    outputs = model(videos)
                else:
                    with torch.no_grad():
                        outputs = model(videos)
                _, preds = torch.max(outputs.data, 1)

                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if mode == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()
                # print('Current Loss {:.4f}'.format(loss.item()))
                running_correct += torch.sum(preds == labels.data).item()
            epoch_loss = running_loss / data_size[mode]
            epoch_acc = running_correct / float(data_size[mode]*BATCH_SIZE)

            print('{} Loss {:.4f}  Accuracy {:.4f}'.format(mode, epoch_loss, epoch_acc))
            # record the epoch
            if mode == 'train':
              if epoch == 1:
                  vis.line (X=np.array ([epoch]), Y=np.array ([epoch_loss]),
                            win=train_loss, update='replace')
                  vis.line (X=np.array ([epoch]), Y=np.array ([epoch_acc]),
                            win=train_acc, update='replace')
              else:
                  vis.line(X=np.array([epoch]), Y=np.array([epoch_loss]),
                       win=train_loss, update='append')
                  vis.line(X=np.array([epoch]), Y=np.array([epoch_acc]),
                       win=train_acc, update='append')

              with open('trainingRecord.txt','a') as f:
                 f.write('epoch {} {} Loss: {:.4f} Acc: {:.4f}\n'.format(
                           epoch, mode, epoch_loss, epoch_acc))
            if mode == 'val':
                if epoch == 1:
                    vis.line (X=np.array ([epoch]), Y=np.array ([epoch_loss]),
                              win=val_loss, update='replace')
                    vis.line (X=np.array ([epoch]), Y=np.array ([epoch_acc]),
                          win=val_acc, update='replace')
                else:
                    vis.line (X=np.array ([epoch]), Y=np.array ([epoch_loss]),
                              win=val_loss, update='append')
                    vis.line (X=np.array ([epoch]), Y=np.array ([epoch_acc]),
                          win=val_acc, update='append')
                with open('valRecord.txt','a') as f:
                  f.write('epoch {} {} Loss: {:.4f} Acc: {:.4f}\n'.format(
                           epoch, mode, epoch_loss, epoch_acc))

            # save model
            if mode == 'val' and best_acc < epoch_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), 'model.pkl')


        time_elapsed = time.time () - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

def read_json(filename):
    with open(filename, 'r') as file:
        info = json.load(file)
    return  info


if __name__ == '__main__':

    # data_transform
    data_transforms = {
        'train': transforms.Compose([
            #transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'val': transforms.Compose([
            #transforms.Resize(256),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])}

    # Load the data
    print('Loading the data...')
    data_dir = 'data/'
    video_datasets = {x: videoDataSet(video_info_list=read_json(data_dir+x+'.json'),
                                      transform=data_transforms[x]) for x in ['train', 'val']}
    # wrap your data and label into Tensor
    data_loaders = {x: torch.utils.data.DataLoader(video_datasets[x],
                                                   batch_size=BATCH_SIZE,
                                                   shuffle=True,
                                                   num_workers=10) for x in ['train', 'val']}
    print('Starting to training...')
    data_size = {x: len(data_loaders[x]) for x in ['train', 'val']}
    print('The size of training set is {}'.format(data_size['train']))
    print('The size of val set is {}'.format(data_size['val']))

    # use gpu or not
    use_gpu = torch.cuda.is_available()

    # Load the model
    model = stnet.stnet50(input_channels=3, num_classes=101, T=7, N=5)
    model=nn.DataParallel(model)
    if use_gpu:
        model=model.cuda()

    now_time = datetime.datetime.now()
    now_time=datetime.datetime.strftime(now_time,'%Y-%m-%d %H:%M:%S')
    with open('trainingRecord.txt','a') as f:
          f.write(now_time)
          f.write('\n')
    with open('valRecord.txt','a') as f:
          f.write(now_time)
          f.write('\n')
          f.write('\n')
    # define loss function
    criterion = nn.CrossEntropyLoss ()
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=50, gamma=0.1)
    train_model(model=model,
                criterion=criterion,
                optimizer=optimizer_ft,
                scheduler=exp_lr_scheduler,
                num_epochs=200)
