## FindTheGap

This package provides tools for geometric data analysis, targeted at finding gaps and valleys in data distribution. It provides a (twice-differentiable) density estimator (Quartic Kernel Density Estimator) relying on pytorch for auto-differentaition, and methods to estimate critical points in the density as well as various statistics to identify and trace `gaps' and valleys in the distribution. 

This package can be installed through pip (https://pypi.org/project/findthegap/):

```
pip install findthegap 
```


Dependencies:
* numpy >= 1.19.5

* torch >= 1.10.1

* scipy >= 1.5.4


Notebook requirements:
sklearn, matplotlib


The folder 'examples' contains a notebook showcasing how to use those tools on 2D data (available in the folder data). 

Disclaimer: this code is work in progress and might go through some changes especially for higher (>2!) dimension... 


Contributors: Gabriella Contardo (CCA at Simons Foundation), David W. Hogg(CCA/NYU/MPIA), Jason S.A. Hunt (CCA)