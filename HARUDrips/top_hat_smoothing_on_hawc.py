import collections
import multiprocessing as mp
from pathlib import Path
# from typing import TypeAlias as T

import astropy.units as u
import boost_histogram as bh
import h5py
import healpy as hp
import numpy as np
import uproot as up
from numba import njit
from numpy.typing import NDArray
from scipy.interpolate import interp1d

# ndarray: T = NDArray[np.float64]
# ndint: T = NDArray[np.int64]
# ndarrbool: T = NDArray[np.bool_]
# ndarrvec: T = NDArray[np.float64]


def read_data(bin_id: str, maptree_dict: up.ReadOnlyDirectory):# -> tuple[str, ndarray]:
    """Utility function to apply to each eleemtn for multiprocessing of the maptree

    :param bin_id: analysis bin id
    :param maptree_dict: uproot directory containing the maptree
    :return: tuple of bin_id and the maptree array
    """
    return bin_id, maptree_dict.array(library="np")  # type: ignore


def load_maptree(
    maptree_file_path: str, data_to_retrieve: str = "data", processes: int = 2
):# -> collections.OrderedDict[str, ndarray]:
    """Load the maptree data from the maptree file

    :param maptree_file_path: path to the maptree file
    :param data_to_retrieve: data to retrieve from the maptree, default is "data"
    :param processes: number of processes to use for multiprocessing
    :return: dictionary of the maptree data
    """
    with (
        mp.Pool(processes=processes) as pool,
        up.open(maptree_file_path, handler=up.MemmapSource) as iomaptree,  # type: ignore
    ):
        analysis_bins = iomaptree["BinInfo/name"].array(library="np").astype(str)  # type: ignore

        input_args = [
            (bin_id, iomaptree[f"nHit{bin_id}/{data_to_retrieve}/count"])
            for bin_id in analysis_bins
        ]

        return collections.OrderedDict(pool.starmap(read_data, input_args))


def load_maptree_daily_combined(
    maptree_file_path: str, data_to_retrieve: str = "data"
):# -> collections.OrderedDict[str, ndarray]:
    """Load the maptree data from an HDF5 file

    :param maptree_file_path: path to the maptree file
    :param data_to_retrieve: skymap to recover 'data', 'bkg' or 'exposure', defaults to "data"
    :return: all the available bin skymaps in the maptree for the 'data' requested.
    """
    organized_data: collections.OrderedDict[str, ndarray] = collections.OrderedDict()
    with h5py.File(maptree_file_path, "r") as iohd5:
        for bin_id, bin_data in iohd5.items():
            organized_data[bin_id] = bin_data[data_to_retrieve][:]
        # for bin_id in iohd5.keys():
        #     organized_data[bin_id] = iohd5[bin_id][data_to_retrieve][:]

    return organized_data


def get_quantiles(
    psf_hist: bh.Histogram, quantile: float | list[float]
):# -> float | ndarray:
    """Compute the quantiles of a PSF histogram, the inputs can be a single number or
    a list of desired quantiles. If a list is provided, the function will return an
    array of quantiles. Otherwise, a single number will be returned.

    :param psf_hist: PSF histogram
    :param quantile: desired quantile(s)
    :raises ValueError: raised if the dimensions of the histogram are higher than 1
    :return: location of the quantile(s) in the histogram
    """
    if psf_hist.ndim != 1:
        raise ValueError("Histogram must be 1D")

    pdf = psf_hist.values() / psf_hist.values().sum()
    cdf = np.cumsum(pdf)

    # refined the interpolation to get the exact quantile
    interpolated_quantile = interp1d(psf_hist.axes[0].centers, cdf)
    interp_xvals: ndarray = np.linspace(
        psf_hist.axes[0].centers[0], psf_hist.axes[0].centers[-1], 500
    )
    if isinstance(quantile, list):
        xquant: ndarray = np.zeros(len(quantile), dtype=np.float64)
        for i, q in enumerate(quantile):
            xquant_idx = np.argmin(abs(interpolated_quantile(interp_xvals) - q))
            xquant[i] = interp_xvals[xquant_idx]

        return xquant

    xquant: int = np.argmin(np.abs(interpolated_quantile(interp_xvals) - quantile))

    return interp_xvals[xquant]


def get_declination_bin(
    lower_edges: ndarray, declination: ndarray | float
):# -> ndint | int:
    """Get the declination bin for a given pixel declination

    :param lower_edges: Lower edges of the declination bins
    :param declination: declination of the pixel
    :return: declination bin(s)
    """
    declination_bins_for_all_pixels = (
        np.searchsorted(lower_edges, declination, side="left") - 1
    )

    declination_bins_for_all_pixels[declination_bins_for_all_pixels < 0] = 0

    return declination_bins_for_all_pixels


def obtain_containment(
    response_file_path: str, quantile: float | list[float] = 0.68
):# -> tuple[dict[str, ndarray], ndarray, ndarray]:
    """Get the containment radius from the PSF response file for a given quantile

    :param response_file_path: path to the PSF response file
    :param quantile: quantile for the containment radius
    :return: dictionary of the containment radii for each analysis bin, upper and
    lower edges of the declination bins
    """

    if not Path(response_file_path).exists():
        raise FileNotFoundError(f"File {response_file_path} does not exist")

    with up.open(response_file_path, handler=up.MemmapSource) as response:  # type: ignore
        upper_edges: ndarray = response["DecBins/upperEdge"].array(library="np")  # type: ignore
        lower_edges: ndarray = response["DecBins/lowerEdge"].array(library="np")  # type: ignore

        hawc_dec_bins: list[int] = list(range(upper_edges.size))
        analysis_bins = response["AnalysisBins/name"].array(library="np").astype(str)  # type: ignore

        response_bins_containment = collections.OrderedDict(
            {bin_id: np.empty_like(upper_edges) for bin_id in analysis_bins}
        )

        for bin_id in analysis_bins:
            for dec_bin in hawc_dec_bins:
                psf_hist: bh.Histogram = response[
                    f"dec_{dec_bin:02d}/nh_{bin_id}/PSF_dec{dec_bin}_nh{bin_id}"
                ].to_boost()  # type: ignore
                xquant = get_quantiles(psf_hist, quantile)
                response_bins_containment[bin_id][dec_bin] = xquant

        return response_bins_containment, upper_edges, lower_edges


def generate_filter(
    nside: int,
    ra: float | u.Quantity,
    dec: float | u.Quantity,
    radius: float | u.Quantity,
    inclusive: bool = False,
):# -> ndarrbool:
    """Generate a filter for the pixels within a region of interest

    :param nside: HEALPix nside for map
    :param ra: ra of the region of interest (J2000)
    :param dec: dec of the region of interest (J2000)
    :param radius: radius of the region of interest
    :param inclusive: whether to include pixels whose centers do not lie within the region of
    interest
    :return: filter for the pixels within the region of interest
    """
    if isinstance(ra, u.Quantity):
        ra = ra.to(u.degree)  # type: ignore
    if isinstance(dec, u.Quantity):
        dec = dec.to(u.degree)  # type: ignore
    if isinstance(radius, u.Quantity):
        radius = radius.to(u.radian).value  # type: ignore

    theta = (90 * u.deg - dec).to(u.radian).value  # type: ignore
    phi = ra.to(u.radian).value  # type: ignore

    position_vec = hp.ang2vec(theta, phi)
    roi = hp.query_disc(nside, position_vec, radius, inclusive=inclusive)

    original_pixel_map = np.arange(hp.nside2npix(nside))

    return np.isin(original_pixel_map, roi)


@njit("float64(float64[:], int64[:])", nogil=True)
def sum_pixels_within_roi(hpx_map: ndarray, roi: ndint):# -> float:
    """Assign to each pixel of the map the sum of the pixels within the region of interest

    :param hpx_map: Skymap to be modified
    :param roi: pixels within the region of interest
    :return: sum of the pixels within the region of interest
    """
    return hpx_map[roi].sum()


def get_roi_pixels(
    nside: int, position_vecs: ndarray, size: float, inclusive: bool = False
):# -> ndint:
    """Get the pixels within a region of interest

    :param nside: HEALPix nside for map
    :param position_vecs: cartesian vectors for the active pixels
    :param size: size of the region of interest
    :param inclusive: whether to include pixels whose centers do not lie within the region of
    interest
    :return: pixels within the region of interest
    """
    return hp.query_disc(nside, position_vecs, size, inclusive=inclusive)


def modify_data_per_bin(
    bin_id: str,
    nside: int,
    original_map: ndarray,
    pixel_map: ndint,
    position_vecs: ndarray,
    containment_radii: ndarray,
    outmap: ndarray,
):# -> tuple[str, ndarray]:
    """Utility function to apply to each element for multiprocessing of the maptree

    :param bin_id: analysi bin id from original maptree
    :param nside: HEALPix nside
    :param original_map: HEALPix map to be smoothed
    :param pixel_map: active pixels over which to iterate
    :param position_vecs: cartesian vectors for the active pixels
    :param containment_radii: containment radii for the PSF
    :param outmap: output map
    :return: tuple of bin_id and the smoothed map
    """
    return bin_id, smooth_map(
        nside, original_map, pixel_map, position_vecs, containment_radii, outmap
    )


def smooth_map(
    nside: int,
    original_map: ndarray,
    pixel_map: ndint,
    position_vecs: ndarray,
    psf_containment_radii: ndarray,
    outmap: ndarray,
) -> ndarray:
    """Perform a top hat smoothing over a skymap

    :param nside: HEALPix nside
    :param original_map: HEALPix map to be smoothed
    :param pixel_map: active pixels over which to iterate
    :param position_vecs: cartesian vectors for the active pixels
    :param psf_containment_radii: containment radii for the PSF
    :param outmap: output map
    :return: smoothed map with the same shape as the original map
    """

    # for pix_id, vec, radius in zip(pixel_map, position_vecs, psf_containment_radii):
    #     this_roi = get_roi_pixels(nside, vec, radius, inclusive=False)
    #     outmap[pix_id] = sum_pixels_within_roi(original_map, this_roi)
    # return outmap
    nsplit = 4
    subsets = np.array_split(pixel_map, nsplit)

    filters: list[ndarrbool] = [np.isin(pixel_map, subset) for subset in subsets]

    input_args = [
        (
            nside,
            original_map,
            pixel_map,
            position_vecs,
            psf_containment_radii,
            outmap,
            filter,
        )
        for filter in filters
    ]

    with mp.Pool(processes=nsplit) as pool:
        results = list(pool.starmap(process_subset_map, input_args))

    return np.sum(results, axis=0)


def process_subset_map(
    nside: int,
    original_map: ndarray,
    pixel_map: ndint,
    position_vecs: ndarray,
    psf_containment_radii: ndarray,
    outmap: ndarray,
    filter: ndarrbool,
) -> ndarray:
    """Perform a top hat smoothing over a skymap

    :param nside: HEALPix nside
    :param original_map: HEALPix map to be smoothed
    :param pixel_map: active pixels over which to iterate
    :param position_vecs: cartesian vectors for the active pixels
    :param psf_containment_radii: containment radii for the PSF
    :param outmap: output map
    :return: smoothed map with the same shape as the original map
    """

    pixel_map = pixel_map[filter]
    position_vecs = position_vecs[filter, :]
    psf_containment_radii = psf_containment_radii[filter]

    for pix_id, vec, radius in zip(pixel_map, position_vecs, psf_containment_radii):
        this_roi = get_roi_pixels(nside, vec, radius, inclusive=False)
        outmap[pix_id] = sum_pixels_within_roi(original_map, this_roi)

    return outmap


def output_maptree_data(
    nside: int, original_dict_keys: list[str]
) -> collections.OrderedDict[str, ndarray]:
    """Create an output dictionary with the same keys as the input dictionary

    :param nside: HEALPix nside of map
    :param original_dict_keys: keys of the original dictionary
    :return: output dictionary with the same keys as the input dictionary
    """
    return collections.OrderedDict(
        {bin_id: np.zeros(hp.nside2npix(nside)) for bin_id in original_dict_keys}
    )


def modify_all_data(
    nside: int,
    pixel_map: ndint,
    position_vecs: ndarray,
    containment_radii: collections.OrderedDict[str, ndarray],
    maptree_data: collections.OrderedDict[str, ndarray],
    active_planes: list[str] | None = None,
    filter: ndarrbool | None = None,
    processes: int = 2,
) -> collections.OrderedDict[str, ndarray]:
    """Operate over the entire data set to smooth the maps

    :param nside: HEALPix nside of skymaps
    :param pixel_map: pixel array over which to iterate
    :param position_vecs: cartesian vectors for the active pixels
    :param containment_radii: containment radii for the PSF
    :param maptree_data: dictionary of the maptree data
    :param filter: filter for the pixels within the region of interest
    :param processes: number of processes to use for multiprocessing
    :return: dictionary of the smoothed maps
    """
    import concurrent.futures as cf

    good_planes = list(maptree_data.keys()) if active_planes is None else active_planes

    out_data = output_maptree_data(nside, good_planes)

    if filter is not None:
        pixel_map = pixel_map[filter]
        position_vecs = position_vecs[filter, :]

        containment_radii = collections.OrderedDict(
            {
                bin_id: radii_vals[filter]
                for bin_id, radii_vals in containment_radii.items()
            }
        )

    input_args = [
        (
            bin_id,
            nside,
            maptree_data[bin_id],
            pixel_map,
            position_vecs,
            containment_radii[bin_id],
            out_data[bin_id],
        )
        for bin_id in good_planes
    ]

    # use ProcessPoolExecutor to handle the management of process pools
    with cf.ProcessPoolExecutor(max_workers=processes) as pool:
        if filter is not None:
            results = collections.OrderedDict(
                pool.map(distribute_args_with_filter, input_args)
            )

        else:
            results = collections.OrderedDict(pool.map(distribute_args, input_args))

    return results


def distribute_args_with_filter(args):
    """Proxy function to distribute the arguments for the multiprocessing pool for
    a small region in the sky.

    :param args: parameters corresponding to `modify_data_per_bin` method
    :return: tuple of analysis bin id and smoothed map
    """
    return modify_data_per_bin(*args)


def distribute_args(args):
    """Proxy function for the multiprocessing pool for the entire sky. The arguments are
    the same as `modify_data_per_bin`, but the `process_data_subset` method splits the data
    to enable multiprocessing for faster performance.

    :param args: parameters corresponding to `process_data_subset` method
    :return: tuple of analysis bin id and smoothed map
    """
    return process_data_subset(*args)


def process_data_subset(
    bin_id: str,
    nside: int,
    original_map: ndarray,
    pixel_map: ndint,
    position_vecs: ndarray,
    containment_radii: ndarray,
    outmap: ndarray,
):
    """Process the whole sky by splitting the whole sky into subsets for multiprocessing
    and combine the result afterwards

    :param bin_id: analysis bin id
    :param nside: HEALPix nside of skymap
    :param original_map: original skymap (before smoothing)
    :param pixel_map: HEALPix pixel ids corresponding to the whole sky
    :param position_vecs: cartesian vectors for the active pixels
    :param containment_radii: containment radii for the PSF at given declination
    :param outmap: output map
    :return: tuple of analysis bin id and smoothed map
    """
    nsplit = 3
    subsets = np.array_split(pixel_map, nsplit)
    filters: list[ndarrbool] = [np.isin(pixel_map, subset) for subset in subsets]

    input_args = [
        (
            nside,
            original_map,
            pixel_map,
            position_vecs,
            containment_radii,
            outmap,
            this_filter,
        )
        for this_filter in filters
    ]

    # NOTE: This code is under development and needs more of a deep dive to understand
    # its full functionality
    # Objective: use the pool to process to generate the second parallelization
    # layer with multiprocessing pool which has better control over individual processes
    with mp.Pool(nsplit) as pool:
        results = pool.starmap(smooth_map_alt, input_args)

    return bin_id, np.sum(results, axis=0)


def smooth_map_alt(
    nside: int,
    original_map: ndarray,
    pixel_map: ndint,
    position_vecs: ndarray,
    psf_containment_radii: ndarray,
    outmap: ndarray,
    filter: ndarrbool,
) -> ndarray:
    """Perform a top hat smoothing over a skymap

    :param nside: HEALPix nside
    :param original_map: HEALPix map to be smoothed
    :param pixel_map: active pixels over which to iterate
    :param position_vecs: cartesian vectors for the active pixels
    :param psf_containment_radii: containment radii for the PSF
    :param outmap: output map
    :return: smoothed map with the same shape as the original map
    """

    if filter is not None:
        pixel_map = pixel_map[filter]
        position_vecs = position_vecs[filter, :]
        psf_containment_radii = psf_containment_radii[filter]

    for pix_id, vec, radius in zip(pixel_map, position_vecs, psf_containment_radii):
        this_roi = get_roi_pixels(nside, vec, radius, inclusive=False)
        outmap[pix_id] = sum_pixels_within_roi(original_map, this_roi)

    return outmap
