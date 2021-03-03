from easydict import EasyDict as edict

cfg = edict()
HyperParameters = edict()

### Data locations
cfg.data = '/home/skh259/LinLab/LinLab/MLforAlzheimers/data/ADNI/genetics/data_prep/'
cfg.filtered_ttest = '/home/skh259/LinLab/LinLab/MLforAlzheimers/data/ADNI/genetics/stats/ttest_p_filtered_genes.csv'
cfg.ttest = '/home/skh259/LinLab/LinLab/MLforAlzheimers/data/ADNI/genetics/stats/t_test_geneExpr.csv'
### Storage for results 
cfg.results = '/home/skh259/LinLab/LinLab/MLforAlzheimers/ADNI/gene_expression/results/'

### Storage for plots  
cfg.plots = '/home/skh259/LinLab/LinLab/MLforAlzheimers/ADNI/gene_expression/plots/'

#Number of times to repeat experiments for
cfg.repeat = 10

##HyperParameters
HyperParameters.features = [200,400,600,800,1000] #number of gene features to be selected
HyperParameters.classifier = ['RandomForest', 'GradientBoosting']
HyperParameters.estimators = [50,100,300,500,700,1000]
HyperParameters.classes = ['CN_MCI', 'CN_AD', 'MCI_AD']

HyperParameters.params = [HyperParameters.features,HyperParameters.classifier,HyperParameters.estimators,HyperParameters.classes]





