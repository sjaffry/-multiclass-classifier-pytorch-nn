# From a single notebook implementation to scaling on Sagemaker
## The Notebooks
There are two notebooks in this repo:  
- pytorch-nn-multiclass-classifier.ipynb
- pytorch-nn-multiclass-classifier-sagemaker.ipynb

In these two notebooks, I show the typical journey take by data scientists to quickly build and train an ML model and then the subsequent steps required to scale the build/train process. The first notebook (pytorch-nn-multiclass-classifier.ipynb) builds & trains a feed forward Pytorch neural net all within the notebook. This is the fastest way to build a model but requires the notebook running on a GPU based instance. 

The second noteboook shows the steps required to refactor the code in the first notebook to leverage Sagemaker's scalable and on-demand training infrastructure. 

## The Model  
### multiclass-classifier-pytorch-nn  

Pytorch feed forward neural network for multi-class classification of music data that has representations of the varying intensities of different instrument within a song to help represent it's genre. The instrument intensities in the training dataset are represented as an array of float32 for each sample. The same is expected by the model for prediction of the genre

## Training and test data  
### Download & upload to your S3 bucket
*(You will use this data to create training datasets in the Notebooks)*
### [Training data file](https://d3k4zua8ad0xio.cloudfront.net/pytorch-multiclass/data/DL1_Train.pkl)
### [Test data file](https://d3k4zua8ad0xio.cloudfront.net/pytorch-multiclass/data/DL1_Test.pkl)
