# Privacy in spatial data with high resolution and time invariance

This is a repository for code companioning the paper "Privacy in spatial data with high resolution and time invariance" written by [Andreas Bjerre-Nielsen](https://abjer.github.io/) and [Mikkel Høst Gandil](https://mikkelgandil.github.io/).

Throughout we are using the Danish squarenet (kvadratnettet). A documentation of this is found in  Styrelsen For Dataforsyning and Effektivisering's [documentation](http://www.sdfe.dk/media/gst/65230/kvadratnettet.pdf).

For installation use the following lines of code in your terminal/command prompt to install with conda
```
conda create -n privacy_spatial
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install tqdm statsmodels seaborn scipy scikit-learn jupyter nose pytables networkx jupyter python-louvain geopandas requests -y
ipython kernel install --name privacy_spatial 
```


### Overview

The code for this project consists of three steps. We have used Anaconda Python 3 with packages installed from the conda-forge channel.

***Step 1:*** This step consists in preprocessing the administrative spatial data. If you do not have data from Statistics Denmark, you can use the dummy data generated in Step 0.

***Step 2:*** This next step finds partition candidates for making a final partition. In this step we apply the functions found in the subfolder `sqr`.

***Step 3:*** This final step selects the best partitions among the candidates and put them together into one final partition. We also analyze the output performing various tests of the reliability.

### License
This project is released under the MIT License, which basically lets you do anything you want with our code as long as you provide attribution back to us and don’t hold us liable.
