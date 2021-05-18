#!/usr/bin/env python3
# coding: utf-8
import numpy as np
from napari.qt.threading import thread_worker
from magicgui import magic_factory, widgets
import napari
from napari.types import LabelsData
from typing_extensions import Annotated
import cc3d

@magic_factory
def make_clean_binary_segmentation(
    viewer: "napari.viewer.Viewer",
    input: LabelsData,
    percentile: Annotated[int, {"min": 0, "max": 100, "step": 1}]=95
):
    from napari.qt import thread_worker

    pbar = widgets.ProgressBar()
    pbar.range = (0, 0)  # unknown duration
    make_clean_binary_segmentation.insert(0, pbar)  # add progress bar to the top of widget

    def _add_data(return_value, self=make_clean_binary_segmentation):
        data, kwargs = return_value
        viewer.add_labels(data, **kwargs)
        self.pop(0).hide()  # remove the progress bar

    @thread_worker(connect={"returned": _add_data})
    def _clean_binary_segmentation(input: LabelsData,
                                   percentile: int=95):
        labels_out = cc3d.connected_components(input)
        clean_binary_volume = input.copy()

        num_of_occurences = np.bincount(labels_out.flatten())
        threshold = np.percentile(num_of_occurences, percentile)

        elements_below_thresh = num_of_occurences < threshold
        idx_below_thresh = np.where(elements_below_thresh)[0]

        for cc in range(len(idx_below_thresh)):
            clean_binary_volume[np.where(labels_out == idx_below_thresh[cc])] = 0

        kwargs = dict(
            name='clean_log_segmentation'
        )
        return (clean_binary_volume, kwargs)

    # start the thread
    _clean_binary_segmentation(input=input,
                               percentile=percentile)
