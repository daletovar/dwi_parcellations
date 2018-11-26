# dwi_parcellations
Python scripts for generating connectivity-based parcellations of diffusion data with fsl

For this analysis I used kmeans clustering on the medial prefrontal cortex. The data I used is free-to-use data from the Rockland Sample from the Enhanced Nathan Kline Institute.

This repository contains a folder with preprocessing scripts, a folder with code for analysis/visualization, and a folder holding some of the necessary data. 

Preprocessing:
  - This folder contains a partial processing pipeline for the rockland diffusion sample. It's currently missing a script to generate the rdti files necessary for registration. Additionally, the bedpostx command needs to run before running probtrackx.
  
Clustering:
  - Probtrackx will produce a .mat file, which we will use for clustering. These files are very large and transforming them into sparse matrices will take a long time to run.
  
Data:
  - This folder contains the nifti files needed to run registration and probtrackx and 5 nifti files representing different kmeans clustering solutions.
