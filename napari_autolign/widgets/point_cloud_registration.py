#!/usr/bin/env python3
# coding: utf-8
from probreg import cpd, bcpd, callbacks
import numpy as np
import open3d as o3
import transforms3d as t3d
import time
from napari.types import PointsData
from magicgui import magic_factory, widgets
import napari
from typing_extensions import Annotated

class RegistrationProgressCallback(object):
    def __init__(self, maxiter):
        self.counter = 0
        self.maxiter = maxiter

    def __call__(self, *args):
        self.counter += 1
        print('{}/{}'.format(self.counter, self.maxiter))

def prepare_source_and_target_nonrigid_3d(source_array,
                                          target_array,
                                          voxel_size=5,
                                          every_k_points=2):
    source = o3.geometry.PointCloud()
    target = o3.geometry.PointCloud()
    source.points = o3.utility.Vector3dVector(source_array)
    target.points = o3.utility.Vector3dVector(target_array)
    source = source.uniform_down_sample(every_k_points=every_k_points)
    target = target.uniform_down_sample(every_k_points=every_k_points)
    source = source.voxel_down_sample(voxel_size=voxel_size)
    target = target.voxel_down_sample(voxel_size=voxel_size)
    return source, target

# Add choice of registratoon method and advanced settings
@magic_factory
def make_point_cloud_registration(
    viewer: "napari.viewer.Viewer",
    moving: PointsData,
    fixed: PointsData,
    algorithm: Annotated[str, {"choices": ["BCPD", "Rigid CPD", "Affine CPD", "RANSAC"]}]='BCPD',
    voxel_size: Annotated[int, {"min": 1, "max": 1000, "step": 1}] = 5,
    every_k_points: Annotated[int, {"min": 1, "max": 1000, "step": 1}] = 1,
    max_iterations: Annotated[int, {"min": 1, "max": 1000, "step": 1}] = 50,
    visualise: bool=False):

    from napari.qt import thread_worker

    pbar = widgets.ProgressBar()
    pbar.range = (0, 0)  # unknown duration
    make_point_cloud_registration.insert(0, pbar)  # add progress bar to the top of widget

    # this function will be called after we return
    def _add_data(return_value, self=make_point_cloud_registration):
        moving, fixed, transformed = return_value
        viewer.add_points(moving,
                          name='moving_points',
                          size=5,
                          face_color='red')
        viewer.add_points(fixed,
                          name='fixed_points',
                          size=5,
                          face_color='green')
        viewer.add_points(transformed,
                          name='transformed_points',
                          face_color='blue',
                          size=5)
        self.pop(0).hide()  # remove the progress bar

    @thread_worker(connect={"returned": _add_data})
    def _point_cloud_registration(moving: PointsData,
                                  fixed: PointsData,
                                  algorithm: str='BCPD',
                                  voxel_size: int=5,
                                  every_k_points: int=1,
                                  max_iterations: int=50,
                                  visualise: bool=False):
        start = time.time()
        source, target = prepare_source_and_target_nonrigid_3d(moving,
                                                               fixed,
                                                               voxel_size=voxel_size,
                                                               every_k_points=every_k_points)
        cbs = []
        cbs.append(RegistrationProgressCallback(max_iterations))
        if visualise:
            cbs.append(callbacks.Open3dVisualizerCallback(source, target))

        if algorithm == 'BCPD':
            tf_param = bcpd.registration_bcpd(source,
                                              target,
                                              maxiter=max_iterations,
                                              callbacks=cbs)

        elif algorithm == 'Rigid CPD':
            tf_param, __, __ = cpd.registration_cpd(source,
                                                    target,
                                                    tf_type_name='rigid',
                                                    maxiter=max_iterations,
                                                    callbacks=cbs)

        elif algorithm == 'Affine CPD':
            tf_param, __, __ = cpd.registration_cpd(source,
                                                    target,
                                                    tf_type_name='affine',
                                                    maxiter=max_iterations,
                                                    callbacks=cbs)
        elif algorithm == 'RANSAC':
            raise NotImplementedError

        elapsed = time.time() - start
        print("time: ", elapsed)
        if algorithm == 'BCPD':
            print("result: ", np.rad2deg(t3d.euler.mat2euler(tf_param.rigid_trans.rot)),
                  tf_param.rigid_trans.scale, tf_param.rigid_trans.t, tf_param.v)
        elif algorithm == 'Rigid CPD' or algorithm == 'Affine CPD':
            print("result: ", tf_param)

        return (np.asarray(source.points),
                np.asarray(target.points),
                tf_param._transform(source.points))

    _point_cloud_registration(moving=moving,
                              fixed=fixed,
                              algorithm=algorithm,
                              voxel_size=voxel_size,
                              every_k_points=every_k_points,
                              max_iterations=max_iterations,
                              visualise=visualise)
