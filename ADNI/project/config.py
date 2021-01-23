import torch
from easydict import EasyDict as edict

cfg = edict()


### Data locations
#cfg.data = '/Users/subashkhanal/Desktop/MLforAlzheimers/data/ADNI/genetics/data_prep/#ADNI_ADgenes_DX.csv'
cfg.data = '/Users/subashkhanal/Desktop/MLforAlzheimers/data/ADNI/genetics/data_prep/CN_AD.csv'
#cfg.data = '/Users/subashkhanal/Desktop/MLforAlzheimers/data/ADNI/genetics/data_prep/CN_MCI.csv'
#cfg.data = '/Users/subashkhanal/Desktop/MLforAlzheimers/data/ADNI/genetics/data_prep/MCI_AD.csv'


### Storage for results 
cfg.results = '/Users/subashkhanal/Desktop/MLforAlzheimers/ADNI/results/'

### Storage for plots  
cfg.plots = '/Users/subashkhanal/Desktop/MLforAlzheimers/ADNI/plots/'
