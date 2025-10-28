# Image Catalog Search



## Getting started

Git Repo for Basic Source Seed Detection using Image Processing

You need to [galactic_plane_scan.py](galactic_plane_scan.py) file which loades in the significance fits files of your region and creates the final seeds of your ROI. You just need to run this and yes it does more than just the galactic plane scan! 

In the [galactic_plane_scan.py](galactic_plane_scan.py) file there are  multiple arguments
- [ ] Required arguments:
- [ ] "-M", "--map", "Significance Map file in root format" 
- [ ] --ROI-center ROI Center of the image (ra, dec) or (l, b)
- [ ] --coordsys : Image Coordinate: 'G', 'C'.  (Default: 'G') 
- [ ] "--size" : ROI Size for the image(Default: 5 x 5 degrees)
- [ ] Optional arguments:
- [ ] plot4HWC, plotlhaaso, plotHGPS, plotFermi(WiP), plotSNR, plotPulsar(WiP) are for plotting respective catalogs
- [ ] plotPDF : save all plots in a PDF
- [ ] saveModel : saveModel for final fits
- [ ] plotSurface : plot the surface intensity peak maps
- [ ] plotStackSignif : Plot Change in Significance Ranges

```
NOTE: 
- [ ] Works with default HAL. Might need to install scikit-image package
- [ ] For more details refer to:
- [1] https://private.hawc-observatory.org/wiki/images/8/86/Hawc_im_process.pdf
- [2] https://private.hawc-observatory.org/wiki/images/0/02/DA21January2025_Rishi.pdf

```

If you there are comments/questions/suggestion on the running the script, contact Rishi Babu (rbabu@icecube.wisc.edu) or Ian Herzog (herzogia@msu.edu) or Dan Salazar (salaza82@msu.edu) or Palmer Wentworth (itomura@msu.edu)

