Run MATLAB code hsvid_main.m from folder
It will expect to find ../hsvideo_public_data folder, which must contain folder raw and PSF. raw contains raw measurements, PSF contains calibration. 

The code will execture, and when done, save results in ../hsvideo_public_data/recons/[filename_with_datestamp]

This depends on https://github.com/antipa/proxMin for solving the optimization problem, but any optimization that allows proximal projection can be used. 
