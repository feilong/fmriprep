#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from __future__ import print_function, division, absolute_import, unicode_literals


import numpy as np
import nibabel as nb
from scipy.interpolate import interpn
from datetime import datetime as dt

from builtins import object, str, bytes

from nipype import logging
LOGGER = logging.getLogger('interfaces')


class BSplineFieldmap(object):
    """
    A fieldmap representation object using BSpline basis
    """

    def __init__(self, fmapnii, weights=None, knots_zooms=None, padding=3,
                 pe_dir=1, njobs=-1):

        self._pedir = pe_dir
        if knots_zooms is None:
            knots_zooms = [40., 40., 18.]
            knots_zooms[pe_dir] = 60.

        if not isinstance(knots_zooms, (list, tuple)):
            knots_zooms = [knots_zooms] * 3

        self._knots_zooms = np.array(knots_zooms)

        if isinstance(fmapnii, (str, bytes)):
            fmapnii = nb.as_closest_canonical(nb.load(fmapnii))

        self._fmapnii = fmapnii
        self._padding = padding

        # Pad data with zeros
        self._data = np.zeros(tuple(np.array(
            self._fmapnii.get_data().shape) + 2 * padding))

        # The list of ijk coordinates
        self._fmapijk = get_ijk(self._data)

        # Find padding coordinates
        self._data[padding:-padding,
                   padding:-padding,
                   padding:-padding] = 1
        self._frameijk = self._data[tuple(self._fmapijk.T)] > 0

        # Set data
        self._data[padding:-padding,
                   padding:-padding,
                   padding:-padding] = fmapnii.get_data()

        # Get ijk in homogeneous coords
        ijk_h = np.hstack((self._fmapijk, np.array([1.0] * len(self._fmapijk))[..., np.newaxis]))

        # The list of xyz coordinates
        self._fmapaff = compute_affine(self._data, self._fmapnii.header.get_zooms())
        self._fmapxyz = self._fmapaff.dot(ijk_h.T)[:3, :].T

        # Mask coordinates
        self._weights = self._set_weights(weights)

        # Generate control points
        self._generate_knots()

        self._X = None
        self._coeff = None
        self._smoothed = None

        self._Xinv = None
        self._inverted = None
        self._invcoeff = None

        self._njobs = njobs

    def _generate_knots(self):
        extent = self._fmapaff[:3, :3].dot(self._data.shape[:3])
        self._knots_shape = (np.ceil(
            (extent - self._knots_zooms) / self._knots_zooms) + 3).astype(int)
        self._knots_grid = np.zeros(tuple(self._knots_shape), dtype=np.float32)
        self._knots_aff = compute_affine(self._knots_grid, self._knots_zooms)

        self._knots_ijk = get_ijk(self._knots_grid)
        knots_ijk_h = np.hstack((self._knots_ijk, np.array(
            [1.0] * len(self._knots_ijk))[..., np.newaxis]))  # In homogeneous coords

        # The list of xyz coordinates
        self._knots_xyz = self._knots_aff.dot(knots_ijk_h.T)[:3, :].T

    def _set_weights(self, weights=None):
        if weights is not None:
            extweights = np.ones_like(self._data)
            extweights[self._padding:-self._padding,
                       self._padding:-self._padding,
                       self._padding:-self._padding] = weights

            return extweights[tuple(self._fmapijk.T)]
        return np.ones((len(self._fmapxyz)))

    def _evaluate_bspline(self):
        """ Calculates the design matrix """
        print('[%s] Evaluating tensor-product cubic BSpline on %d points, %d control points' %
              (dt.now(), len(self._fmapxyz), len(self._knots_xyz)))
        self._X = tbspl_eval(self._fmapxyz, self._knots_xyz, self._knots_zooms, njobs=self._njobs)
        print('[%s] Finished BSpline evaluation' % dt.now())

    def fit(self):
        self._evaluate_bspline()

        fieldata = self._data[tuple(self._fmapijk.T)]

        print('[%s] Starting least-squares fitting using %d unmasked points' %
              (dt.now(), len(fieldata[self._weights > 0.0])))
        self._coeff = np.linalg.lstsq(
            self._X[self._weights > 0.0, ...].toarray(),
            fieldata[self._weights > 0.0])[0]
        print('[%s] Finished least-squares fitting' % dt.now())

    def get_coeffmap(self):
        self._knots_grid[tuple(self._knots_ijk.T)] = self._coeff
        return nb.Nifti1Image(self._knots_grid, self._knots_aff, None)

    def get_smoothed(self):
        self._smoothed = np.zeros_like(self._data)
        coords = tuple(self._fmapijk[self._frameijk].T)
        self._smoothed[coords] = self._X[self._frameijk].dot(self._coeff)

        output_image = self._smoothed[self._padding:-self._padding,
                                      self._padding:-self._padding,
                                      self._padding:-self._padding]
        return nb.Nifti1Image(output_image, self._fmapnii.affine, self._fmapnii.header)

    def invert(self):
        targets = self._fmapxyz.copy()
        targets[:, self._pedir] += self._smoothed[tuple(self._fmapijk.T)]
        print('[%s] Inverting transform :: evaluating tensor-product cubic BSpline on %d points, %d control points' %
              (dt.now(), len(targets), len(self._knots_xyz)))
        self._Xinv = tbspl_eval(targets, self._knots_xyz, self._knots_zooms, self._njobs)
        print('[%s] Finished BSpline evaluation, %s' %
              (dt.now(), str(self._Xinv.shape)))

        print('[%s] Starting least-squares fitting using %d unmasked points' %
              (dt.now(), len(targets)))
        self._invcoeff = np.linalg.lstsq(
            self._Xinv, self._fmapxyz[:, self._pedir] - targets[:, self._pedir])[0]
        print('[%s] Finished least-squares fitting' % dt.now())

    def get_inverted(self):
        self._inverted = np.zeros_like(self._data)
        self._inverted[tuple(self._fmapijk.T)] = self._X.dot(self._invcoeff)
        return nb.Nifti1Image(self._inverted, self._fmapnii.affine, self._fmapnii.header)

    def interp(self, in_data, inverse=False, fwd_pe=True):
        dshape = tuple(in_data.shape)
        gridxyz = self._fmapxyz.reshape((dshape[0], dshape[1], dshape[2], -1))

        x = gridxyz[:, 0, 0, 0]
        y = gridxyz[0, :, 0, 1]
        z = gridxyz[0, 0, :, 2]

        xyzmin = (x.min(), y.min(), z.min())
        xyzmax = (x.max(), y.max(), z.max())

        targets = self._fmapxyz.copy()

        if inverse:
            factor = 1.0 if fwd_pe else -1.0
            targets[:, self._pedir] += factor * \
                self._inverted[tuple(self._fmapijk.T)]
        else:
            targets[:, self._pedir] += self._smoothed[tuple(self._fmapijk.T)]

        interpolated = np.zeros_like(self._data)
        interpolated[tuple(self._fmapijk.T)] = interpn(
            (x, y, z), in_data, [tuple(v) for v in targets],
            bounds_error=False, fill_value=0)

        return nb.Nifti1Image(interpolated, self._fmapnii.affine, self._fmapnii.header)

#     def xfm_coords(self, in_coord):
#         X = fif.tbspl_eval(np.array([in_coord]), self._knots_xyz, self._knots_zooms)
#         new_coord = in_coord + X.dot(self._coeff if not inverse else self._invcoeff)

def get_ijk(data, offset=0):
    """
    Calculates voxel coordinates from data
    """
    from numpy import mgrid

    if not isinstance(offset, (list, tuple)):
        offset = [offset] * 3

    grid = mgrid[offset[0]:(offset[0] + data.shape[0]),
                 offset[1]:(offset[1] + data.shape[1]),
                 offset[2]:(offset[2] + data.shape[2])]
    return grid.reshape(3, -1).T

def compute_affine(data, zooms):
    """
    Compose a RAS affine mat, since the affine of the image might not be RAS
    """
    aff = np.eye(4) * (list(zooms) + [1])
    aff[:3, 3] -= aff[:3, :3].dot(np.array(data.shape[:3], dtype=float) - 1.0) * 0.5
    return aff

def _approx(fmapnii, s=14.):
    """
    Slice-wise approximation of a smooth 2D bspline
    credits: http://scipython.com/book/chapter-8-scipy/examples/two-dimensional-interpolation-\
    with-scipyinterpolaterectbivariatespline/

    """
    from scipy.interpolate import RectBivariateSpline
    from builtins import str, bytes

    if isinstance(fmapnii, (str, bytes)):
        fmapnii = nb.load(fmapnii)

    if not isinstance(s, (tuple, list)):
        s = np.array([s] * 2)

    data = fmapnii.get_data()
    zooms = fmapnii.header.get_zooms()

    knot_decimate = np.floor(s / np.array(zooms)[:2]).astype(np.uint8)
    knot_space = np.array(zooms)[:2] * knot_decimate

    xmax = 0.5 * data.shape[0] * zooms[0]
    ymax = 0.5 * data.shape[1] * zooms[1]

    x = np.arange(-xmax, xmax, knot_space[0])
    y = np.arange(-ymax, ymax, knot_space[1])

    x2 = np.arange(-xmax, xmax, zooms[0])
    y2 = np.arange(-ymax, ymax, zooms[1])

    coeffs = []
    nslices = data.shape[-1]
    for k in range(nslices):
        data2d = data[..., k]
        data2dsubs = data2d[::knot_decimate[0], ::knot_decimate[1]]
        interp_spline = RectBivariateSpline(x, y, data2dsubs)

        data[..., k] = interp_spline(x2, y2)
        coeffs.append(interp_spline.get_coeffs().reshape(data2dsubs.shape))

    # Save smoothed data
    hdr = fmapnii.header.copy()
    caff = fmapnii.affine
    datanii = nb.Nifti1Image(data.astype(np.float32), caff, hdr)

    # Save bspline coeffs
    caff[0, 0] = knot_space[0]
    caff[1, 1] = knot_space[1]
    coeffnii = nb.Nifti1Image(np.stack(coeffs, axis=2), caff, hdr)

    return datanii, coeffnii


def bspl_smoothing(fmapnii, masknii=None, knot_space=[18., 18., 20.]):
    """
    A 3D BSpline smoothing of the fieldmap
    """
    from datetime import datetime as dt
    from builtins import str, bytes
    from scipy.linalg import pinv2

    if not isinstance(knot_space, (list, tuple)):
        knot_space = [knot_space] * 3
    knot_space = np.array(knot_space)

    if isinstance(fmapnii, (str, bytes)):
        fmapnii = nb.load(fmapnii)

    data = fmapnii.get_data()
    zooms = fmapnii.header.get_zooms()

    # Calculate hi-res i
    ijk = np.where(data < np.inf)
    xyz = np.array(ijk).T * np.array(zooms)[np.newaxis, :3]

    # Calculate control points
    xyz_max = xyz.max(axis=0)
    knot_dims = np.ceil(xyz_max / knot_space) + 2
    bspl_grid = np.zeros(tuple(knot_dims.astype(int)))
    bspl_ijk = np.where(bspl_grid == 0)
    bspl_xyz = np.array(bspl_ijk).T * knot_space[np.newaxis, ...]
    bspl_max = bspl_xyz.max(axis=0)
    bspl_xyz -= 0.5 * (bspl_max - xyz_max)[np.newaxis, ...]

    points_ijk = ijk
    points_xyz = xyz

    # Mask if provided
    if masknii is not None:
        if isinstance(masknii, (str, bytes)):
            masknii = nb.load(masknii)
        data[masknii.get_data() <= 0] = 0
        points_ijk = np.where(masknii.get_data() > 0)
        points_xyz = np.array(points_ijk).T * np.array(zooms)[np.newaxis, :3]

    print('[%s] Evaluating tensor-product cubic-bspline on %d points' %
          (dt.now(), len(points_xyz)))
    # Calculate design matrix
    X = tbspl_eval(points_xyz, bspl_xyz, knot_space)
    print('[%s] Finished, bspline grid has %d control points' %
          (dt.now(), len(bspl_xyz)))
    Y = data[points_ijk]

    # Fit coefficients
    print('[%s] Starting least-squares fitting' % dt.now())
    # coeff = (pinv2(X.T.dot(X)).dot(X.T)).dot(Y) # manual way (seems equally
    # slow)
    coeff = np.linalg.lstsq(X, Y)[0]
    print('[%s] Finished least-squares fitting' % dt.now())
    bspl_grid[bspl_ijk] = coeff
    aff = np.eye(4)
    aff[:3, :3] = aff[:3, :3] * knot_space[..., np.newaxis]
    coeffnii = nb.Nifti1Image(bspl_grid, aff, None)

    # Calculate hi-res design matrix:
    # print('[%s] Evaluating tensor-product cubic-bspline on %d points' % (dt.now(), len(xyz)))
    # Xinterp = tbspl_eval(xyz, bspl_xyz, knot_space)
    # print('[%s] Finished, start interpolation' % dt.now())

    # And interpolate
    newdata = np.zeros_like(data)
    newdata[points_ijk] = X.dot(coeff)
    newnii = nb.Nifti1Image(newdata, fmapnii.affine, fmapnii.header)

    return newnii, coeffnii


def tbspl_eval(points, knots, zooms, njobs=None):
    """
    Evaluate tensor product BSpline
    """
    from scipy.sparse import vstack
    from fmriprep.utils.maths import bspl

    points = np.array(points, dtype=float)
    knots = np.array(knots, dtype=float)
    vbspl = np.vectorize(bspl)

    if njobs is not None and njobs < 1:
        njobs = None

    if njobs == 1:
        coeffs = [_evalp((p, knots, vbspl, zooms)) for p in points]
    else:
        from multiprocessing import Pool
        pool = Pool(processes=njobs, maxtasksperchild=100)
        coeffs = pool.map(
            _evalp, [(p, knots, vbspl, zooms) for p in points])
        pool.close()
        pool.join()

    return vstack(coeffs)

def _evalp(args):
    import numpy as np
    from scipy.sparse import csr_matrix

    point, knots, vbspl, zooms = args
    u_vec = (knots - point[np.newaxis, ...]) / zooms[np.newaxis, ...]
    c = vbspl(u_vec.reshape(-1)).reshape((knots.shape[0], 3)).prod(axis=1)
    return csr_matrix(c)
