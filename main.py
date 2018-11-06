import os
import glob
from itertools import chain

IMG_DIR = './train-dataset'

dnames = glob.glob('{}/*'.format(IMG_DIR))

fnames = [glob.glob('{}/*.png'.format(d)) for d in dnames
          if not os.path.exists('{}/ignore'.format(d))]
fnames = list(chain.from_iterable(fnames))

