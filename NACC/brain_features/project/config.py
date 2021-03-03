from easydict import EasyDict as edict

cfg = edict()
HyperParameters = edict()

### Data locations
cfg.data = '/home/skh259/LinLab/LinLab/MLforAlzheimers/data/NACC/lin10232020mri_diag.csv'


### Storage for results 
cfg.results = '/home/skh259/LinLab/LinLab/MLforAlzheimers/NACC/brain_features/results/'

### Storage for plots  
cfg.plots = '/home/skh259/LinLab/LinLab/MLforAlzheimers/NACC/brain_features/plots/'

#Number of times to repeat experiments for
cfg.repeat = 10

##HyperParameters
HyperParameters.features = ['full'] #full or partial features selected #only 155 features so full as default
HyperParameters.classifier = ['RandomForest', 'GradientBoosting']
HyperParameters.estimators = [50,100,300,500,700,1000]
HyperParameters.classes  = ['CN_EMCI', 'CN_LMCI', 'CN_AD', 'EMCI_LMCI','EMCI_AD', 'LMCI_AD']

HyperParameters.params = [HyperParameters.features,HyperParameters.classifier,HyperParameters.estimators,HyperParameters.classes]

