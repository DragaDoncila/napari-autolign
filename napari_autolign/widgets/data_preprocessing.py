#!/usr/bin/env python3
# coding: utf-8
import numpy as np
from magicgui import magic_factory, widgets
from scipy import ndimage
from napari.types import ImageData

@magic_factory
def make_data_preprocessing(
    viewer: "napari.viewer.Viewer",
    input: ImageData,
    input_xy_pixelsize: float,
    input_z_pixelsize: float,
    reference_xy_pixelsize: float,
    reference_z_pixelsize: float
):
    from napari.qt import thread_worker

    pbar = widgets.ProgressBar()
    pbar.range = (0, 0)  # unknown duration
    make_data_preprocessing.insert(0, pbar)  # add progress bar to the top of widget

    def _add_data(return_value, self=make_data_preprocessing):
        print('Adding new layer')
        data, kwargs = return_value
        viewer.add_image(data, **kwargs)
        self.pop(0).hide()  # remove the progress bar

    def _zoom_values(xy, z, xy_ref, z_ref):
        xy_zoom = xy / xy_ref
        z_zoom = z / z_ref

        return xy_zoom, z_zoom

    @thread_worker(connect={"returned": _add_data})
    def _preprocess(input: ImageData,
                    input_xy_pixelsize: float,
                    input_z_pixelsize: float,
                    reference_xy_pixelsize: float,
                    reference_z_pixelsize: float):
        xy_zoom, z_zoom = _zoom_values(input_xy_pixelsize,
                                       input_z_pixelsize,
                                       reference_xy_pixelsize,
                                       reference_z_pixelsize)
        print('Zoom values', xy_zoom, z_zoom)
        output = ndimage.zoom(input, (z_zoom, xy_zoom, xy_zoom))
        print('Zoomed image')
        kwargs = dict(
            name='preprocessed_image'
        )
        return (output, kwargs)

    _preprocess(input=input,
                input_xy_pixelsize=input_xy_pixelsize,
                input_z_pixelsize=input_z_pixelsize,
                reference_xy_pixelsize=reference_xy_pixelsize,
                reference_z_pixelsize=reference_z_pixelsize)
