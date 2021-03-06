# Privacy in spatial data with high resolution and time invariance

This is a repository for code companioning the paper "Privacy in spatial data with high resolution and time invariance" written by [Andreas Bjerre-Nielsen](https://abjer.github.io/) and [Mikkel Høst Gandil](https://mikkelgandil.github.io/).

Throughout we are using the Danish squarenet (kvadratnettet). A documentation of this is found in  Styrelsen For Dataforsyning and Effektivisering's [documentation](http://www.sdfe.dk/media/gst/65230/kvadratnettet.pdf).

### Overview

The code for this project consists of three steps. We have used Anaconda Python 3 with packages installed from the conda-forge channel.

***Step 1:*** This step consists in preprocessing the administrative spatial data. Before proceeding with this step you need to download and generate the Danish square net grid data, see [Grid Factory](http://www.routeware.dk/download.php). You also need to contact Statistics Denmark to get the population in each square net cell.

***Step 2:*** This next step finds partition candidates for making a final partition. In this step we apply the functions found in the subfolder `sqr`.

***Step 3:*** This final step selects the best partitions among the candidates and put them together into one final partition. We also analyze the output performing various tests of the reliability.

### License
This project is released under the MIT License, which basically lets you do anything you want with our code as long as you provide attribution back to us and don’t hold us liable.
