This is the project for statistical and Machine Learning based experiments for different projects utilizing the ADNI database.
http://adni.loni.usc.edu/data-samples/access-data/

The goal is to use multi-modal data from genetics, neuropsychological assesments, different modalities of Medical imaging (MRI, PET) to build better predictive models. 
In this direction, the gene expression data (which is the measure of expression of genes assessed through blood samples of subjects) is considered to be used with MRI images. Owing to large numbe of genes in human genome, there are large gene probes at which the gene expression is evaluated. With low sample size, building effective Machine Learning models using such high dimensional data is challenging. Therefore several feature selection techniques should be use. As a easy first step, currently following to approaches have been tried for feature selection.
1. Perform a student's ttest between the desired groups and select top N gene probes which are the most statistically significant locations. Here N=200 is picked for now and the results labeled as "full" are the classification performance for these 200 genetic features. 
2. Refer the literature specifically this: (https://core.ac.uk/download/pdf/206791729.pdf) and use the probes corresponding to the genes that are reported to have some association with Alzheimer's disease.


