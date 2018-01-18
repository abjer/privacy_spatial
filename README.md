# Privacy in spatial data with high resolution and time invariance

This is a repository for code companioning the paper "Privacy in spatial data with high resolution and time invariance" written by [Andreas Bjerre-Nielsen](abjer.github.io) and [Mikkel Høst Gandil](https://mikkelgandil.github.io/).

Throughout we are using the Danish squarenet (kvadratnettet). A documentation of this is found in  Styrelsen For Dataforsyning and Effektivisering's [documentation](http://www.sdfe.dk/media/gst/65230/kvadratnettet.pdf).

### Overview

The code for this project consists of three steps.

***Step 0:*** This step consists in preprocessing the administrative spatial data. Before proceeding with this step you need to download generate the Danish square net grid data, see [Grid Factory](http://www.routeware.dk/download.php). You also need to contact Statistics Denmark for getting the population in each square net cell.

***Step 1:*** This next step finds partition candidates for making a final partition. In this step we apply the functions found in the subfolder `sqr`.

***Step 2:*** This final step consists in finding we choose the best partitions among the candidates and put them together into one final partition. We also analyze the output performing various tests and the reliability.

### License
This project is released under the MIT License, which basically lets you do anything you want with our code as long as you provide attribution back to us and don’t hold us liable.
