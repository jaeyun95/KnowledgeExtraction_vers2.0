## This is knowledge extraction system version 2.0


# Setting up Environment
```
wget https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh
conda update -n base -c defaults conda
conda create --name kes python=3.6 pytorch=1.0.1
source activate kes
conda install numpy h5py transformers
conda install cudatoolkit=9.0 -c pytorch
pip install allennlp==0.8.0
pip install sentence-transformers
```