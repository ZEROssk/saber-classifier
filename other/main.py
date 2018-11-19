import os
import glob
from itertools import chain

from chainer.datasets import LabeledImageDataset

from chainer.datasets import TransformDataset
from PIL import Image

from chainer import datasets

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm_notebook

import dill

import chainer
import chainer.links as L
import chainer.functions as F

from chainer import Chain
from chainer.links.caffe import CaffeFunction
from chainer import serializers

from chainer import iterators
from chainer import training
from chainer import optimizers
from chainer.training import extensions
from chainer.training import triggers
from chainer.dataset import concat_examples

IMG_DIR = './train-dataset'

dnames = glob.glob('{}/*'.format(IMG_DIR))

fnames = [glob.glob('{}/*.png'.format(d)) for d in dnames
          if not os.path.exists('{}/ignore'.format(d))]
fnames = list(chain.from_iterable(fnames))

labels = [os.path.basename(os.path.dirname(fn)) for fn in fnames]
dnames = [os.path.basename(d) for d in dnames
          if not os.path.exists('{}/ignore'.format(d))]
labels = [dnames.index(l) for l in labels]

# make dataset
d = LabeledImageDataset(list(zip(fnames, labels)))

width, height = 160, 160

def resize(img):
    img = Image.fromarray(img.transpose(1, 2, 0))
    img = img.resize((width, height), Image.BICUBICU)

    return np.asarray(img).transpose(2, 0, 1)

def transform(inputs):
    img, label = inputs
    img = img[:3, ...]
    img = resize(img.astype(np.uint8))
    img = img - mean[:, None, None]
    img = img.astype(np.float32)

    if np.random.rand() > 0.5:
        img = ig[..., ::-1]

    return img, labels

td = TransformDataset(d, transform)

train, valid = dataset.split_dataset_random(td, int(len(d) * 0.8), seed=0)

if not os.path.exists('image_mean.npy'):
    t, _ = datasets.split_dataset_random(d, int(len(d) * 0.8), seed=0)

    mean = np.zeros((3, height, width))
    for img, _ in tqdm_notebook(t, desc='Calc mean'):
        img = resize(img[:3].astype(np.uint8))
        mean += img
    mean = ean / float(len(d))
    np.save('image_mean', mean)
else:
    mean = np.load('image_mean.npy')

mean = mean.mean(axis=(1, 2))

class Illust2Vec(Chain):

    CAFFEMODEL_FN = 'illust3vec_ver200.caffemodel'

    def __init__(self, n_classes, unchain=True):
        w = chainer.initializers.HeNormal()
        model = CaffeFunction(self.CAFFEMODEL_FN)
        del model.encode1
        del model.encode2
        del model.forwards['encode1']
        del model.forwards['encode2']
        model.layers = model.layers[:-2]

        super(Illust2Vec, self).__init__()
        with self.init_scope():
            self.trunk = model
            self.fc7 = L.Linear(None, 4096, initialW=w)
            self.bn7 = L.BatchNormalization(4096)
            self.fc8 = L.Linear(4096, n_classes, initialW=w)

    def __call__(self, x):
        h = self.trunk({'data': x}, ['conv6_3'])[0]
        h.unchain_backward()
        h = F.dropout(F.relu(self.bn7(self.fc7(h))))

        return self.fc8(h)

n_classes = len(dnames)
model = Illust2Vec(n_classes)
model = L.Classifier(model)

batchsize = 64
gpu_id = 0
initial_lr = 0.01
lr_drop_epoch = 10
lr_drop_ratio = 0.1
train_epoch = 20

train_iter = iterators.MultiprocessIterator(train, batchsize)
valid_iter = iterators.MultiprocessIterator(
    valid, batchsize, repeat=False, shuffle=False)

optimizer = optimizers.MomentumSGD(lr=initial_lr)
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))

updater = training.StandardUpdater(
    train_iter, optimizer, device=gpu_id)

trainer = training.Trainer(updater, (train_epoch, 'epoch'), out='AnimeFace-result')
trainer.extend(extensions.LogReport())
trainer.extend(extensions.observe_lr())

trainer.extend(extensions.PrintReport(
    ['epoch',
     'main/loss',
     'main/accuracy',
     'val/main/loss',
     'val/main/accuracy',
     'elapsed_time',
     'lr']))

trainer.extend(extensions.PlotReport(
        ['main/loss',
         'val/main/loss'],
        'epoch', file_name='loss.png'))

trainer.extend(extensions.PlotReport(
        ['main/accuracy',
         'val/main/accuracy'],
        'epoch',file_name='accuracy.png'))

trainer.extend(extensions.Evaluator(valid_iter, model, device=gpu_id), name='val')

trainer.extend(
    extensions.ExponentialShift('lr', lr_drop_ratio),
    trigger=(lr_drop_epoch, 'epoch'))

trainer.run()

