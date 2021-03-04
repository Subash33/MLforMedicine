"""
This is just to parse the results from messy log output produced by MLforAlzhiemer's code
@Subash Khanal March 4, 2021
"""
import os
import pandas as pd
#filename = '/home/skh259/LinLab/LinLab/logs/ADNI_ML_gene_exp/ADNI_gene_exp_hypSweep.log'
#outpath = '/home/skh259/LinLab/LinLab/MLforAlzheimers/ADNI/gene_expression/results/'

filename = '/home/skh259/LinLab/LinLab/logs/NACC_ML_brain/NACC_brain_hypSweep.log'
outpath = '/home/skh259/LinLab/LinLab/MLforAlzheimers/NACC/brain_features/results/'
features= []
classifier = []
estimator = []
classes = []
acc = []
f1 = []

with open(filename,'r') as infile:
    txt = infile.read()
    t = txt.strip().split('For parameters:')
    for l in range(1,len(t)):

        op = t[l].strip().split('[[')[0].replace('\n',' ').replace('(','').replace(')','').replace(',', '').strip().split(' ')
        #print(op)
        if 'NACC' in filename:
            features.append(op[0])
        if 'ADNI' in filename:
            features.append(int(op[0]))   
        classifier.append(op[1])
        estimator.append(int(op[2]))
        classes.append(op[3])
        acc.append(float(op[4]))
        f1.append(float(op[5]))

myDict = {'features':features,
            'classifier':classifier,
            'estimator':estimator,
            'classes':classes,
            'acc':acc,
            'f1':f1
}

df = pd.DataFrame(myDict)
csv_file = filename.split('.')[0].split('/')[-1]+'.csv'
output_file = os.path.join(outpath,csv_file)
df.to_csv(output_file)
