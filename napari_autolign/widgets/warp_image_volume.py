#!/usr/bin/env python3
# coding: utf-8
# Adapted from: https://github.com/zpincus/celltool/blob/master/celltool/numerics/image_warp.py

from scipy import ndimage
import numpy as np
from probreg import bcpd
import napari
from magicgui import magic_factory, widgets
from napari.types import PointsData, ImageData
from typing_extensions import Annotated
import math
from typing import Sequence
from pathlib import Path
from napari.qt import thread_worker

def _warp_images(from_points, to_points, image, output_region, interpolation_order=5, approximate_grid=10):
    print('Entered warp_images')
    transform = _make_inverse_warp(from_points, to_points, output_region, approximate_grid)
    return ndimage.map_coordinates(np.asarray(image), transform, order=interpolation_order)

def _make_inverse_warp(from_points, to_points, output_region, approximate_grid):
    x_min, y_min, z_min, x_max, y_max, z_max = output_region
    if approximate_grid is None: approximate_grid = 1
    x_steps = (x_max - x_min) // approximate_grid
    y_steps = (y_max - y_min) // approximate_grid
    z_steps = (z_max - z_min) // approximate_grid

    x, y, z = np.mgrid[x_min:x_max:x_steps*1j, y_min:y_max:y_steps*1j, z_min:z_max:z_steps*1j]
    transform = _make_warp(to_points, from_points, x, y, z)

    if approximate_grid != 1:
        # linearly interpolate the zoomed transform grid
        new_x, new_y, new_z = np.mgrid[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
        x_fracs, x_indices = np.modf((x_steps-1)*(new_x-x_min)/float(x_max-x_min))
        y_fracs, y_indices = np.modf((y_steps-1)*(new_y-y_min)/float(y_max-y_min))
        z_fracs, z_indices = np.modf((z_steps-1)*(new_z-z_min)/float(z_max-z_min))

        x_indices = x_indices.astype(int)
        y_indices = y_indices.astype(int)
        z_indices = z_indices.astype(int)

        x1 = 1 - x_fracs
        y1 = 1 - y_fracs
        z1 = 1 - z_fracs

        ix1 = (x_indices+1).clip(0, x_steps-1)
        iy1 = (y_indices+1).clip(0, y_steps-1)
        iz1 = (z_indices+1).clip(0, z_steps-1)

        transform_x = _trilinear_interpolation(0, transform, x1, y1, z1, x_fracs, y_fracs, z_fracs, x_indices, y_indices, z_indices, ix1, iy1, iz1)
        transform_y = _trilinear_interpolation(1, transform, x1, y1, z1, x_fracs, y_fracs, z_fracs, x_indices, y_indices, z_indices, ix1, iy1, iz1)
        transform_z = _trilinear_interpolation(2, transform, x1, y1, z1, x_fracs, y_fracs, z_fracs, x_indices, y_indices, z_indices, ix1, iy1, iz1)

        transform = [transform_x, transform_y, transform_z]
    return transform

def _trilinear_interpolation(d, t, x0, y0, z0, x1, y1, z1, ix0, iy0, iz0, ix1, iy1, iz1):
    t000 = t[d][(ix0, iy0, iz0)]
    t001 = t[d][(ix0, iy0, iz1)]
    t010 = t[d][(ix0, iy1, iz0)]
    t100 = t[d][(ix1, iy0, iz0)]
    t011 = t[d][(ix0, iy1, iz1)]
    t101 = t[d][(ix1, iy0, iz1)]
    t110 = t[d][(ix1, iy1, iz0)]
    t111 = t[d][(ix1, iy1, iz1)]

    return t000*x0*y0*z0 + t001*x0*y0*z1 + t010*x0*y1*z0 + t100*x1*y0*z0 + t011*x0*y1*z1 + t101*x1*y0*z1 + t110*x1*y1*z0 + t111*x1*y1*z1

def _U(x):
    _small = 1e-100
    return (x**2) * np.where(x<_small, 0, np.log(x))

def _interpoint_distances(points):
    xd = np.subtract.outer(points[:,0], points[:,0])
    yd = np.subtract.outer(points[:,1], points[:,1])
    zd = np.subtract.outer(points[:,2], points[:,2])
    return np.sqrt(xd**2 + yd**2 + zd**2)

def _make_L_matrix(points):
    n = len(points)
    K = _U(_interpoint_distances(points))
    P = np.ones((n, 4))
    P[:,1:] = points
    O = np.zeros((4, 4))
    L = np.asarray(np.bmat([[K, P],[P.transpose(), O]]))
    return L

def _calculate_f(coeffs, points, x, y, z):
    w = coeffs[:-3]
    a1, ax, ay, az = coeffs[-4:]
    summation = np.zeros(x.shape)
    for wi, Pi in zip(w, points):
        summation += wi * _U(np.sqrt((x-Pi[0])**2 + (y-Pi[1])**2 + (z-Pi[2])**2))
    return a1 + ax*x + ay*y +az*z + summation

def _make_warp(from_points, to_points, x_vals, y_vals, z_vals):
    from_points, to_points = np.asarray(from_points), np.asarray(to_points)
    err = np.seterr(divide='ignore')
    L = _make_L_matrix(from_points)
    V = np.resize(to_points, (len(to_points)+4, 3))
    V[-3:, :] = 0
    coeffs = np.dot(np.linalg.pinv(L), V)
    print('L, V, coeffs', L.shape, V.shape, coeffs.shape)
    x_warp = _calculate_f(coeffs[:,0], from_points, x_vals, y_vals, z_vals)
    y_warp = _calculate_f(coeffs[:,1], from_points, x_vals, y_vals, z_vals)
    z_warp = _calculate_f(coeffs[:,2], from_points, x_vals, y_vals, z_vals)
    np.seterr(**err)
    return [x_warp, y_warp, z_warp]

# TODO: Use scipy.ndimage.rotate() and scipy.ndimage.shift()
# TODO: Access transformation matrix through CSV file
def _warp_image_volume_rigid(image,
                             matrix,
                             output_region,
                             interpolation_order):
    """Wraps scipy.ndimage.rotate() and scipy.ndimage.shift()"""
    raise NotImplementedError

# TODO: Use scipy.ndimage.affine()
def _warp_image_volume_affine(image,
                              matrix,
                              output_region,
                              interpolation_order):
    """Wraps scipy.ndimage.affine()"""
    raise NotImplementedError

def on_init(widget):
    """Initializes widget layout amd updates widget layout according to user input."""

    for x in ['moving_points', 'transformed_points']:
        setattr(getattr(widget, x), 'visible', False)

    def toggle_transform_widget(event):
        if event.value == "Deformable":
            for x in ['moving_points', 'transformed_points']:
                setattr(getattr(widget, x), 'visible', True)
            for x in ['transformation_matrix']:
                setattr(getattr(widget, x), 'visible', False)

        else:
            for x in ['transformation_matrix']:
                setattr(getattr(widget, x), 'visible', True)
            for x in ['moving_points', 'transformed_points']:
                setattr(getattr(widget, x), 'visible', False)

    widget.transform_type.changed.connect(toggle_transform_widget)

@magic_factory(widget_init=on_init, layout='vertical', call_button="Warp")
def make_image_warping(
    viewer: "napari.viewer.Viewer",
    moving_image: ImageData,
    fixed_image: ImageData,
    transform_type: Annotated[str, {"choices": ["Rigid", "Affine", "Deformable"]}],
    moving_points: PointsData,
    transformed_points: PointsData,
    transformation_matrix: Sequence[Path],
    interpolation_order: Annotated[int, {"min": 0, "max": 10, "step": 1}]=1,
    approximate_grid: Annotated[int, {"min": 1, "max": 10, "step": 1}]=1,
    sub_division_factor: Annotated[int, {"min": 1, "max": 10, "step": 1}]=1
):
    pbar = widgets.ProgressBar()
    pbar.range = (0, 0)  # unknown duration
    make_image_warping.insert(0, pbar)  # add progress bar to the top of widget

    # this function will be called after we return
    def _add_data(return_value, self=make_image_warping):
        data, kwargs = return_value
        viewer.add_image(data, **kwargs)
        self.pop(0).hide()  # remove the progress bar

    @thread_worker(connect={"returned": _add_data})
    def _warp_image_volume(moving_image: ImageData,
                           fixed_image: ImageData,
                           moving_points: PointsData,
                           transformed_points: PointsData,
                           interpolation_order: int=1,
                           approximate_grid: int=1,
                           sub_division_factor: int=1):

        print('Warping image volume')
        assert len(moving_points) == len(transformed_points), 'Moving and transformed points must be of same length.'

        x_chunk = math.ceil(fixed_image.shape[1] / sub_division_factor)
        y_chunk = math.ceil(fixed_image.shape[2] / sub_division_factor)
        z_chunk = math.ceil(fixed_image.shape[0] / sub_division_factor)

        warped_image = np.empty(fixed_image.shape)

        for x in range(math.ceil(fixed_image.shape[1] / x_chunk)):
          for y in range(math.ceil(fixed_image.shape[2] / y_chunk)):
            for z in range(math.ceil(fixed_image.shape[0] / z_chunk)):
                output_region = (z * z_chunk,
                                 x * x_chunk,
                                 y * y_chunk,
                                 min((z * z_chunk + z_chunk), fixed_image.shape[0]),
                                 min((x * x_chunk + x_chunk), fixed_image.shape[1]),
                                 min((y * y_chunk + y_chunk), fixed_image.shape[2]))

                z_min, x_min, y_min, z_max, x_max, y_max = output_region

                print('Output region:', output_region)

                warped_region = _warp_images(from_points=moving_points,
                                             to_points=transformed_points,
                                             image=moving_image,
                                             output_region=output_region,
                                             interpolation_order=interpolation_order,
                                             approximate_grid=approximate_grid)

                print('Warped region:', warped_region.shape)
                print('Warped region smaller:', warped_region[:-1, :-1, :-1].shape)

                # Warping function returns images padded by one in each dimension
                warped_image[z_min:z_max, x_min:x_max, y_min:y_max] = warped_region[:-1, :-1, :-1]

        kwargs = dict(
            name='warped_image'
        )
        print('Finished warping')
        return (warped_image, kwargs)

    if transform_type == 'Deformable':
        _warp_image_volume(moving_image=moving_image,
                           fixed_image=fixed_image,
                           moving_points=moving_points,
                           transformed_points=transformed_points,
                           interpolation_order=interpolation_order,
                           approximate_grid=approximate_grid,
                           sub_division_factor=sub_division_factor)

    elif transform_type == 'Rigid':
        _warp_image_volume_rigid()

    elif transform_type == 'Affine':
        _warp_image_volume_affine()
