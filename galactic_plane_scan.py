import time
# python galactic_plane_scan.py -M /Users/rishi/Documents/Analysis/data/aligned-ml-wholesky-neg-ext0.0.fits.gz --size 10 8 --plotPDF True --ROI-center 60 0
start_time = time.time()
import os
import yaml
import re 

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patheffects as pe
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.animation import PillowWriter
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
from matplotlib import patches
from scipy.interpolate import interp1d

import healpy as hp
from reproject import reproject_from_healpix
import json
from astropy.coordinates import ICRS
from astroquery.simbad import Simbad
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import Angle, SkyCoord
import astropy.units as u
from astropy.wcs import utils as asutils
from astropy.visualization.wcsaxes.frame import EllipticalFrame

from scipy.optimize import curve_fit
from skimage.filters import difference_of_gaussians, window, gaussian
from skimage.io import imread, imshow
from skimage.feature import peak_local_max
from skimage import color, exposure, transform
from skimage.feature import blob_dog, blob_log, blob_doh

from scipy.ndimage import gaussian_filter

import argparse as ap
from helpers import *
milagrotextcolor, milagro = setupMilagroColormap(-3, 15, 2, 256)
milagrotextcolor2, milagro2 = setupMilagroColormap(0.2, 1, 2, 256)
current=os.getcwd()
parser = ap.ArgumentParser(
    description="Produce a list of seed sources.",
    formatter_class=ap.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "-M", "--map", default=ap.SUPPRESS, required=True, help="Significance Map file in fits format"
)
parser.add_argument("--ROI-center",action="store",required=True,dest="roiCenter",type=float,nargs=2,default=None,help="ROI Center of the image (ra, dec)/ (l, b)",)
parser.add_argument("--coordsys",action="store",dest="coordsys",default='G',help="Image Coordinate: 'G', 'C'.  (Default: 'G')",)
parser.add_argument("--size",action="store",dest="size",type=float,nargs=2,default=(5, 5),help="ROI Size for the image(Default: 5 x 5 degrees)",)
parser.add_argument("--plot4HWC",action="store",dest="plot4hwc",default=False,help="Overlay 4HWC Catalog Results",)
parser.add_argument("--plotlhaaso",action="store",dest="plotlhaaso",default=False,help="Overlay 1LHAASO Catalog Results",)
parser.add_argument("--plotHGPS",action="store",dest="plothgps",default=False,help="Overlay H.E.S.S. HGPS Catalog Results",)
parser.add_argument("--plotFermi",action="store",dest="plotfermi",default=False,help="Overlay 4FGL Catalog Results (Not Fully Implemented)",)
parser.add_argument("--plotSNR",action="store",dest="plotsnr",default=False,help="Overlay SNR Catalog",)
parser.add_argument("--plotPulsar",action="store",dest="plotpulsar",default=False,help="Overlay Pulsar ATNF Catalog",)
parser.add_argument("--psfsize",action="store",dest="psfsize",type=float,nargs=1,default=0.2,help="Size of the gaussian smearing (Default: 0.2 degrees)",)
parser.add_argument("--plotPDF",action="store",dest="plotPDF",default=False,help="Save all plots to a PDF",)
parser.add_argument("--saveModel", action="store_true", dest="saveModel", help="Save Model for Fitting")
parser.add_argument("--plotSurface",action="store",dest="plotSurface",default=False,help="Plot Intensity Surface Gradient",)
parser.add_argument("--plotStackSignif",action="store",dest="plotStackSignif",default=False,help="Plot Change in Significance",)
parser.add_argument("--colormap",action="store",dest="colormap",default=milagro,help="Colormap (Default: Milagro)",)
args = parser.parse_args()




outdir = os.getcwd()+ '/plots/'
gen_dir = os.getcwd()+'/generated_files'
if not os.path.exists(outdir):
    os.makedirs(outdir)
if not os.path.exists(gen_dir):
    os.makedirs(gen_dir)
map_tree = args.map
roi_ra, roi_dec = args.roiCenter[0], args.roiCenter[1]
is4hwc=args.plot4hwc
ishgps=args.plothgps
islhaaso=args.plotlhaaso
isfermi=args.plotfermi
issnr=args.plotsnr
ispular=args.plotpulsar
psf=args.psfsize
catalogs=[]
if is4hwc:
    catalogs.append('4hwc')
if ishgps:
    catalogs.append('hgps')
if islhaaso:
    catalogs.append('lhaaso')
if isfermi:
    catalogs.append('fermi')
if issnr:
    catalogs.append('snr')
if ispular:
    catalogs.append('pulsar')
else:
    catalogs.append('None')

ispdf=args.plotPDF
if ispdf:
    pdf = PdfPages(os.path.join(outdir, f'run-{roi_ra}-{roi_dec}-plots.pdf')) 
else:
    pdf = None
filename = map_tree

ra_center = roi_ra
dec_center = roi_dec

print(f"ROI center {ra_center}, {dec_center}")

coord_sys = args.coordsys
xlength=args.size[0]
ylength=args.size[1]

origin = [ra_center, dec_center, xlength, ylength] 
ra_center = "{:.2f}".format(ra_center)
dec_center = "{:.2f}".format(dec_center)

array, footprint, wcs = loadmap(filename, coord_sys, origin, 'origin')
xnum = array.shape[1]
ynum = array.shape[0]

pixel_size = wcs.wcs.cdelt[1]
print(f'Degrees per pixel: {pixel_size} ')
print(f'Shape of Map in pixel number: {xnum} X {ynum}')

if ispdf:
    make_plots(array, wcs, pixel_size, coordsys='G', threshold=3, vmin=-5, vmax=6, contour=True, title='SkyMap', hotspots=None, save_dir=outdir, pdf=pdf, cmap='ult', figsize=(10, 6), labels=catalogs)
    find_peak(array, wcs)
    # makeplot(array, -5, 15, wcs, args.colormap, coord_sys, "Gamma-ray Sky", xnum, ynum, outdir, catalogs, pdf=True)
else:
   make_plots(array, wcs, pixel_size, coordsys='G', threshold=3, vmin=-5, vmax=6, contour=True, title='SkyMap', hotspots=None, save_dir=outdir, pdf=None, cmap='ult', figsize=(10, 6), labels=catalogs)

# Stop if there are no hotspots greater than 5 sigma
if np.max(array) < 5:
    print(f'Algorithm wont proceed. Significance Map has no data greater than 4$\sigma$)')
    exit
else:
    print(f'Algorithm will proceed. Significance Map has data greater than 4$\sigma$)')

# Preprocessing Step: Floor the image for analysis
if np.min(array) < -6:
    print("Image softly floored to -6 sigma")
    array = soft_floor(array, floor_min=-6, scale=1.0)

if args.plotSurface:
    # Make the 3D Intensity Distribution
    smoothed = array
    smoothed = smoothed[::4, ::4]
    x = np.arange(smoothed.shape[1])
    y = np.arange(smoothed.shape[0])
    X, Y = np.meshgrid(x, y)

    coordinates = peak_local_max(smoothed, min_distance=5, threshold_rel=0.1)
    peak_x = coordinates[:, 1]
    peak_y = coordinates[:, 0]
    peak_z = smoothed[peak_y, peak_x]


    fig = plt.figure(figsize=(6, 5))#, dpi=80)
    ax = fig.add_subplot(111, projection='3d')
    surface = ax.plot_surface(X, Y, smoothed, cmap='plasma', edgecolor='none')
    peaks_plot = ax.scatter(peak_x, peak_y, peak_z, c='black', s=50, label='Peaks')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Intensity')
    ax.set_title('3D Surface Histogram of Significance')
    def update(frame):
        ax.view_init(elev=30, azim=frame)
        return fig,

    ani = animation.FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=50)
    ani.save(outdir+f'gaussian_surface_rotation-{1825}.gif', writer='pillow')

if args.plotStackSignif:
    #Make Stacked Significane Range Plots
    x = np.linspace(3, np.max(array)+3, 15)
    fig = plt.figure(figsize=(20,8))
    ax = plt.subplot(1, 1, 1, projection=wcs)
    im = ax.imshow(array, cmap=milagro, vmin=x[0], vmax=x[1])
    cbar = fig.colorbar(im, orientation='vertical', fraction=0.046, pad=0.04, label='Significance')
    plot_ax_label(ax, coord_sys)
    ax.set_title(f'HAWC Sky Map from {x[0]} to {x[1]}')
    ax.set_xlim(0, xnum)
    ax.set_ylim(0, ynum)

    def update(i):
        if i < len(x) - 1:
            im.set_clim(vmin=x[i], vmax=x[i+1])
            ax.set_title(f'HAWC Sky Map from {x[i]} to {x[i+1]}')
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(x)-1, interval=1000, blit=False)
    ani.save(outdir+f'hawc_sky_map-{1825}.gif', writer=PillowWriter(fps=1))
    plt.clf()


# Calculate the 1D histogram of the image
pixels = array.flatten()
binlen = int(len(pixels)/1000)
print(f"Length of bins={binlen}")
counts, bin_edges = np.histogram(pixels, bins=binlen, range=(-4, np.max(pixels)))
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 

def gaussian_fit(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean)**2) / (2 * stddev**2))

x_exp = np.linspace(-5, 5, binlen)
y_exp = gaussian_fit(x_exp, counts.max(), 0, 1)

initial_guess = [counts.max(), bin_centers[np.argmax(counts)], np.std(pixels)]
popt, _ = curve_fit(gaussian_fit, bin_centers, counts, p0=initial_guess)
# x_fit = np.linspace(bin_centers[0]-2, bin_centers[-1], 500)
x_fit = np.linspace(-6, np.max(pixels), binlen)
y_fit = gaussian_fit(x_fit, *popt)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
log_counts = np.where(counts > 0, counts, 1) 
ax.hist(pixels, bins=binlen, range=(np.min(pixels), np.max(pixels)), color='fuchsia', edgecolor='green', alpha=0.6, label='Histogram (Counts)', histtype='step', linewidth=3)
ax.plot(x_exp, y_exp, 'red', linewidth=2, label=f'Expectation\nμ=0, σ=0.01')
ax.plot(x_fit, y_fit, 'blue', linewidth=2, label=f'Fit\nμ={popt[1]:.5f}, σ={popt[2]:.5f}')
ax.set_yscale('log')
ax.set_ylim(0.5, 1e6)
ax.set_xlim(-6, 6)
ax.set_xlabel('Pixel Intensity')
ax.set_ylabel('Log(Counts)')
ax.set_title('Histogram of Image')
ax.legend()
ax.minorticks_on()
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
if ispdf:
    pdf.savefig(fig, bbox_inches='tight')
else:
    plt.savefig(outdir+'OriginalImageHistogram.png')
plt.clf()

# Normalize the image for further analysis
image = array
norm_image = (image-np.min(image))/(np.max(image)-np.min(image))

make_plots(norm_image, wcs, pixel_size, coordsys='C', threshold=0.4, vmin=0, vmax=1, contour=True, title='Normalized Image', hotspots=None, save_dir=outdir, pdf=pdf, cmap='ult', figsize=(10, 6), labels=catalogs)
find_peak(array, wcs)

#Smearing Image with a gaussian for visualizing 
image_tmp = norm_image
smear_radius = 0.2
smear_image = gaussian(image_tmp, sigma=smear_radius/pixel_size)

make_plots(smear_image, wcs, pixel_size, coordsys='C', threshold=0.2, vmin=np.min(smear_image), vmax=np.max(smear_image), title=f"Gaussian Smeared Image with radius {smear_radius}$^\degree$", hotspots=None, save_dir=outdir, pdf=pdf, cmap='ult', figsize=(10, 6), labels=catalogs)
find_peak(array, wcs)

# Find the Difference of Gaussian (DoG) Image
psf=0.2             #PSF of the dec band
print(psf/pixel_size)
dog_image = difference_of_gaussians(norm_image, 1, psf/pixel_size)

make_plots(dog_image, wcs, pixel_size, coordsys='G', threshold=0.005, vmin=np.min(dog_image), vmax=np.max(dog_image), title=f'Gaussian Subtracted Image', hotspots=None, save_dir=outdir, pdf=pdf, cmap='ult', figsize=(10, 6), labels=catalogs)

# Calculate the 1D histogram of the DoG Image

pixels = dog_image.flatten()
sigma_resid2 = np.std(dog_image)


counts, bin_edges = np.histogram(pixels, bins=200, range=(np.min(pixels), np.max(pixels)))
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 

def gaussian_fit(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean)**2) / (2 * stddev**2))

x_exp = np.linspace(np.min(bin_edges), np.max(bin_edges), 200)
y_exp = gaussian_fit(x_exp, counts.max(), -0.1, 0.01)

initial_guess = [counts.max(), bin_centers[np.argmax(counts)], np.std(pixels)]
popt, _ = curve_fit(gaussian_fit, bin_centers, counts, p0=initial_guess)
x_fit = np.linspace(bin_centers[0], bin_centers[-1], 500)
y_fit = gaussian_fit(x_fit, *popt)


fit_interp = interp1d(x_fit, y_fit, bounds_error=False, fill_value=0)
gaussian_at_bins = fit_interp(bin_centers)

# Compute deviation mask: where histogram exceeds Gaussian expectation significantly
deviation_mask = (bin_centers > popt[1]) & (counts > 1.1 * gaussian_at_bins)

for i in range(len(deviation_mask)):
    if np.all(deviation_mask[i:]):
        j = i
        break
deviation_location = bin_centers[j]
print(f"Significant deviation from Gaussian on positive side at:{j},  {deviation_location:.5f}")

sigma_resid = popt[2]
deviation = 3*sigma_resid
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)

log_counts = np.where(counts > 0, counts, 1) 
ax.hist(pixels, bins=200, range=(np.min(pixels), np.max(pixels)),  color='fuchsia', edgecolor='green', alpha=0.6, label='Histogram (Counts)',histtype='step', linewidth=3)
ax.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'Fit\nμ={popt[1]:.5f}, σ={popt[2]:.5f}')
if np.any(deviation_mask):
    ax.axvline(deviation_location, color='blue', linestyle='--', linewidth=2, 
                label=f'Deviation from fit ≈ {deviation_location:.5f}')
ax.axvline(deviation, label=f'3 $\sigma$  = {3*popt[2]:.5f}', color='black')
ax.axvline(popt[1], label=f'Mean = {popt[1]:.5f}')
ax.set_yscale('log')
ax.set_ylim(1, 1e8)
ax.set_xlim(np.min(bin_edges), np.max(bin_edges))
ax.set_xlabel('Pixel Intensity')
ax.set_ylabel('Log(Counts)')
ax.set_title('Histogram of Gaussian Subtracted Image')
ax.legend()
ax.minorticks_on()
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
if ispdf:
    pdf.savefig(fig, bbox_inches='tight')
else:
    plt.savefig(outdir+'OriginalImageHistogram.png')
plt.clf()

lowfreq_image=norm_image-dog_image
lowfreq_image = gaussian(lowfreq_image, sigma=3)

# Find the blobs from the DoG image
threshold_val = 0.01
ps_blobs = blob_dog(dog_image, min_sigma=0.1/pixel_size, max_sigma=psf/pixel_size, threshold=threshold_val, exclude_border=20, overlap=0.5)
print("Number of point source blobs=",len(ps_blobs))
ext_blobs = blob_dog(dog_image,min_sigma=psf/pixel_size, max_sigma=0.5/pixel_size, threshold=threshold_val, exclude_border=20, overlap=0.5)
print("Number of ext source blobs=",len(ext_blobs))

ps_blobs = ps_blobs[ps_blobs[:, 2] < psf/pixel_size]
ext_blobs = ext_blobs[ext_blobs[:, 2] > psf/pixel_size]
print("Number of point source blobs=",len(ps_blobs))
print("Number of ext source blobs=",len(ext_blobs))

# DoG Image Intensity Filtering
deviation=deviation_location
intensity_min = deviation
ps_filtered_blobs, ps_filtered_coords, ps_filtered_radius = blob_filter_intensity(ps_blobs, dog_image, intensity_min, wcs, pixel_size)
print("No of ps = ", len(ps_filtered_blobs))
ext_filtered_blobs, ext_filtered_coords, ext_filtered_radius = blob_filter_intensity(ext_blobs, dog_image, intensity_min, wcs, pixel_size)
print("No of ext = ", len(ext_filtered_blobs))



blobs_dict = {
    'psblobs': ps_filtered_blobs,
    'extblobs': ext_filtered_blobs,
}
# make_plots(array, wcs, pixel_size, coordsys='G', blobs=blobs_dict, title=f'', cmap=milagro, labels=['4hawc'])

make_plots(array, wcs, pixel_size, coordsys='G', blobs=blobs_dict, title=f"DoG Results", hotspots=None, save_dir=outdir, pdf=pdf, cmap=milagro, figsize=(10, 6), labels=['4hawc'])

# Input Significance Image Intensity Filtering
ps_filtered_blobs2, ps_filtered_coords2, ps_filtered_radius2 = blob_filter_intensity(ps_filtered_blobs, array,5 , wcs, pixel_size)
ext_filtered_blobs3, ext_filtered_coords3, ext_filtered_radius3 = blob_filter_intensity(ext_filtered_blobs, array, 5, wcs, pixel_size)
print("No of ps = ", len(ps_filtered_blobs2))
print("No of ext = ", len(ext_filtered_blobs3))

blobs_dict = {
    'psblobs': ps_filtered_blobs2,
    'extblobs': ext_filtered_blobs3,
}

make_plots(array, wcs, pixel_size, coordsys='G', blobs=blobs_dict, title=f"DoG First Filter", hotspots=None, save_dir=outdir, pdf=pdf, cmap=milagro, figsize=(10, 6), labels=['4hawc'])


lowfreq_image=norm_image-gaussian(dog_image, sigma=8)
lowfreq_image = gaussian(lowfreq_image, sigma=3)

make_plots(lowfreq_image, wcs, pixel_size, vmin=np.min(lowfreq_image), vmax=np.max(lowfreq_image), threshold=0.02,coordsys='G',title=f'Low Freq Image', cmap='ult', labels=catalogs,save_dir=outdir, pdf=pdf, figsize=(10, 6))


# Find the blobs from the Gaussian Smeared Image
ext_blobs4 = blob_dog(lowfreq_image,min_sigma=0.4/pixel_size, max_sigma=0.8/pixel_size, threshold=0.001, exclude_border=30)#, sigma_ratio=4)
print("Number of ext source blobs=",len(ext_blobs4))

# Filter the Gaussian Blobs from above
ext_filtered_blobs4, ext_filtered_coords4, ext_filtered_radius4 = blob_filter_intensity(ext_blobs4, array, 3, wcs, pixel_size)

blobs_dict = {
    'psblobs': ps_filtered_blobs2,
    'extblobs': ext_filtered_blobs4,
}

make_plots(array, wcs, pixel_size, coordsys='G', blobs=blobs_dict, title=f'Second DoG Detection', cmap='ult', labels=catalogs
, save_dir=outdir, pdf=pdf, figsize=(10, 6))

# Find the blobs from the Gaussian Smeared Image
ext_blobs2 = blob_dog(lowfreq_image,min_sigma=0.8/pixel_size, max_sigma=1.5/pixel_size, threshold=0.005, exclude_border=30)#, sigma_ratio=4)
print("Number of ext source blobs=",len(ext_blobs2))

# Filter the Gaussian Blobs from above
ext_filtered_blobs2, ext_filtered_coords2, ext_filtered_radius2 = blob_filter_intensity(ext_blobs2, array, 4.5, wcs, pixel_size)
temp_ext = ext_filtered_blobs4+ext_filtered_blobs2
blobs_dict = {
    # 'psblobs': ps_filtered_blobs2,
    # 'extblobs': ext_filtered_blobs2,
    'extblobs': temp_ext,
}

make_plots(array, wcs, pixel_size, coordsys='G', blobs=blobs_dict, title=f'Third DoG Detection', cmap='ult', labels=catalogs, save_dir=outdir, pdf=pdf, figsize=(10, 6))

all_coords = ps_filtered_coords2 + ext_filtered_coords2 + ext_filtered_coords3 + ext_filtered_coords4
all_radii = ps_filtered_radius2 + ext_filtered_radius2 + ext_filtered_radius3 + ext_filtered_radius4

filtered_df = filter_overlapping_sources(all_coords, all_radii, radius_to_sigma)

t_coords = ps_filtered_coords2 + ext_filtered_coords2
t_radii = ps_filtered_radius2 + ext_filtered_radius2

# Filter ext group 4
valid_idx = remove_ext_sources_with_radius_overlap(
    t_coords, t_radii,
    ext_filtered_coords4, ext_filtered_radius4
)

# Apply filtering
ext_filtered_coords4 = [ext_filtered_coords4[i] for i in valid_idx]
ext_filtered_radius4 = [ext_filtered_radius4[i] for i in valid_idx]
ext_filtered_blobs4  = [ext_filtered_blobs4[i]  for i in valid_idx]


fig, ax = plt.subplots(figsize=(8, 2))
ax.axis('off') 

try:
    table = ax.table(cellText=filtered_df.values,
                        colLabels=filtered_df.columns,
                        cellLoc='center',
                        loc='center')

    table.scale(1, 2)
except:
    text = f"DRIPS Compilation of Result for Crab Simulation Run. No Source Found"
    ax.text(0.5, 0.5, text, fontsize=12, ha='center', va='center', wrap=True)
if ispdf:
    pdf.savefig(fig, bbox_inches='tight')
else:
    plt.savefig(outdir+'OriginalImageHistogram.png')
plt.close(fig)


# Plot the Final Seeds
seed_final = pd.DataFrame({
    "Name": filtered_df['Names'], 
    "ra": filtered_df['ra'],
    "dec":  filtered_df['dec'], 
    "ext": filtered_df['Sigma Radius'],
    })

blobs_dict = {
    'psblobs': ps_filtered_blobs2,
    'extblobs': ext_filtered_blobs2,
}
make_plots(array, wcs, pixel_size,threshold=5, coordsys='G',hotspots=seed_final, title=f'SkyMap', cmap='ult', labels=['4hawc'], save_dir=outdir, pdf=pdf, figsize=(16, 8))


otype_dict = {
    'SN*': 'SuperNova',
    'SN?': 'SuperNova',
    'XB*': 'X-ray Binary',
    'XB?': 'X-ray Binary',
    'LXB': 'Low Mass X-ray Binary',
    'LXB?': 'Low Mass X-ray Binary',
    'HXB': 'High Mass X-ray Binary',
    'HXB?': 'High Mass X-ray Binary',
    'Psr': 'Pulsar',
    'SFR': 'Star Forming Region',
    'HII': 'HII Region',
    'SNR': 'SuperNova Remnant',
    'SR?': 'SuperNova Remnant',
    'gam': 'Gamma-ray Source',
    'gB': 'Gamma-ray Burst',
    'X': 'X-ray Source',
    'ULX': 'Ultra-luminous X-ray Source',
    'UX?': 'Ultra-luminous X-ray Source',
}
Simbad.add_votable_fields('parallax', 'mesdistance', 'otype') 
Simbad.get_votable_fields()
simbad = Simbad()
simbad.ROW_LIMIT = 0 

list_table = []
sorted_table = []
counterparts = []
snr = []
lmxb = []
hmxb = []
pulsar = []
sfr = []
for i in range(len(filtered_df)):
    source_coord=SkyCoord(ra=filtered_df['ra'][i]*u.degree, dec=filtered_df['dec'][i]*u.degree, frame='icrs')
    source_radius = filtered_df['Sigma Radius'][i] * u.degree
    gal_coords = source_coord.transform_to('galactic')
    result_table = Simbad.query_region(gal_coords, radius=source_radius)
    result_table['distance_pc'] = 1000 / result_table['plx_value'] 
    sorted_table = result_table[result_table['distance_pc'].argsort()]
    df = result_table.to_pandas()
    df_unique = df.drop_duplicates(subset='main_id')
    sorted_df = df_unique.sort_values(by='distance_pc')
    sorted_df['otype_expanded'] = sorted_df['otype'].map(otype_dict).fillna(df['otype'])
    if sorted_df[sorted_df['otype_expanded'] == 'SuperNova Remnant']['main_id'].any():
        snr.append(sorted_df[sorted_df['otype_expanded'] == 'SuperNova Remnant']['main_id'].values)
    else:
        snr.append("None")
    
    if sorted_df[sorted_df['otype_expanded'] == 'Low Mass X-ray Binary']['main_id'].any():
        lmxb.append(sorted_df[sorted_df['otype_expanded'] == 'Low Mass X-ray Binary']['main_id'].values)
    else:
        lmxb.append("None")

    if sorted_df[sorted_df['otype_expanded'] == 'High Mass X-ray Binary']['main_id'].any():
        hmxb.append(sorted_df[sorted_df['otype_expanded'] == 'High Mass X-ray Binary']['main_id'].values)
    else:
        hmxb.append("None")
        
    if sorted_df[sorted_df['otype_expanded'] == 'Pulsar']['main_id'].any():
        pulsar.append(sorted_df[sorted_df['otype_expanded'] == 'Pulsar']['main_id'].values)
    else:
        pulsar.append("None")

    if sorted_df[sorted_df['otype_expanded'] == 'Star Forming Region']['main_id'].any():
        sfr.append(sorted_df[sorted_df['otype_expanded'] == 'Star Forming Region'].values)
    else:
        sfr.append("None")
    
    gammasource =  sorted_df[sorted_df['otype_expanded'] == 'Gamma-ray Source']
    xraysource =  sorted_df[sorted_df['otype_expanded'] == 'X-ray Source']    
counterparts = pd.DataFrame({
    "Name": filtered_df['Names'], 
    "snr": snr,
    "lmxb":  lmxb, 
    "hmxb": hmxb,
    "pulsar":pulsar,
    "sfr": sfr,
    })
with open(outdir+f'out{ra_center}_{dec_center}.txt', 'w') as f:
    print('Filename:', filename, file=f)
    for i in range(len(counterparts)):
        print(f"Source: {counterparts['Name'][i]}", file=f)
        print(f"  SNRs: {counterparts['snr'][i]}", file=f)
        print(f"  LMXBs: {counterparts['lmxb'][i]}", file=f)
        print(f"  HMXBs: {counterparts['hmxb'][i]}", file=f)
        print(f"  Pulsars: {counterparts['pulsar'][i]}", file=f)
        print(f"  SFRs: {counterparts['sfr'][i]}", file=f)
        print("--------------------------------------------------", file=f)


if ispdf:
    pdf.close()