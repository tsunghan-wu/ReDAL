import abc
import random
import logging

import numpy as np
import scipy
import scipy.ndimage
import scipy.interpolate
import torch


class Transform(abc.ABC):

    @abc.abstractmethod
    def __init__(self):
        return NotImplemented

    @abc.abstractmethod
    def __call__(self):
        return NotImplemented


class PointCloudTransform(Transform):
    # In 2D, flip, shear, scale, and rotation of images are coordinate transformation
    pass


class FeatureTransform(Transform):
    # color jitter, hue, etc., are feature transformations
    pass


class VoxelTransform(Transform):
    pass


class Compose(Transform):
    # Support multiple input trasform composition

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args


class RandomApply(Transform):
    # Random apply the transforms given probablility

    def __init__(self, transforms, p=0.5):
        self.p = p
        self.transforms = transforms

    def __call__(self, *args):
        if np.random.random() < self.p:
            for t in self.transforms:
                args = t(*args)
        return args


class ChromaticTranslation(FeatureTransform):
    """Add random color to the image, input must be an array in [0,255] or a PIL image"""

    def __init__(self, trans_range_ratio=1e-1):
        """
        trans_range_ratio: ratio of translation i.e. 255 * 2 * ratio * rand(-0.5, 0.5)
        """
        self.trans_range_ratio = trans_range_ratio

    def __call__(self, coords, feats, labels):
        tr = (np.random.rand(1, 3) - 0.5) * 255 * 2 * self.trans_range_ratio
        feats[:, :3] = np.clip(tr + feats[:, :3], 0, 255)
        return coords, feats, labels


class ChromaticAutoContrast(FeatureTransform):

    def __init__(self, randomize_blend_factor=True, blend_factor=0.5):
        self.randomize_blend_factor = randomize_blend_factor
        self.blend_factor = blend_factor

    def __call__(self, coords, feats, labels):
        '''
        feats: [0-255]
        '''
        # mean = np.mean(feats, 0, keepdims=True)
        # std = np.std(feats, 0, keepdims=True)
        # lo = mean - std
        # hi = mean + std
        lo = feats[:, :3].min(0, keepdims=True)
        hi = feats[:, :3].max(0, keepdims=True)
        assert hi.max() > 1, "invalid color value. Color is supposed to be [0-255]"

        scale = 255 / (hi - lo)

        contrast_feats = (feats[:, :3] - lo) * scale

        blend_factor = random.random() if self.randomize_blend_factor else self.blend_factor
        feats[:, :3] = (1 - blend_factor) * feats + blend_factor * contrast_feats
        return coords, feats, labels


class ChromaticJitter(FeatureTransform):

    def __init__(self, std=0.01):
        self.std = std

    def __call__(self, coords, feats, labels):
        noise = np.random.randn(feats.shape[0], 3)
        noise *= self.std * 255
        feats[:, :3] = np.clip(noise + feats[:, :3], 0, 255)
        return coords, feats, labels


class HueSaturationTranslation(FeatureTransform):

    @staticmethod
    def rgb_to_hsv(rgb):
        # Translated from source of colorsys.rgb_to_hsv
        # r,g,b should be a numpy arrays with values between 0 and 255
        # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
        rgb = rgb.astype('float')
        hsv = np.zeros_like(rgb)
        # in case an RGBA array was passed, just copy the A channel
        hsv[..., 3:] = rgb[..., 3:]
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = np.max(rgb[..., :3], axis=-1)
        minc = np.min(rgb[..., :3], axis=-1)
        hsv[..., 2] = maxc
        mask = maxc != minc
        hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
        rc = np.zeros_like(r)
        gc = np.zeros_like(g)
        bc = np.zeros_like(b)
        rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
        gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
        bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
        hsv[..., 0] = np.select([r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
        hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
        return hsv

    @staticmethod
    def hsv_to_rgb(hsv):
        # Translated from source of colorsys.hsv_to_rgb
        # h,s should be a numpy arrays with values between 0.0 and 1.0
        # v should be a numpy array with values between 0.0 and 255.0
        # hsv_to_rgb returns an array of uints between 0 and 255.
        rgb = np.empty_like(hsv)
        rgb[..., 3:] = hsv[..., 3:]
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        i = (h * 6.0).astype('uint8')
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
        rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
        rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
        rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
        return rgb.astype('uint8')

    def __init__(self, hue_max, saturation_max):
        self.hue_max = hue_max
        self.saturation_max = saturation_max

    def __call__(self, coords, feats, labels):
        # Assume feat[:, :3] is rgb
        hsv = HueSaturationTranslation.rgb_to_hsv(feats[:, :3])
        hue_val = (random.random() - 0.5) * 2 * self.hue_max
        sat_ratio = 1 + (random.random() - 0.5) * 2 * self.saturation_max
        hsv[..., 0] = np.remainder(hue_val + hsv[..., 0] + 1, 1)
        hsv[..., 1] = np.clip(sat_ratio * hsv[..., 1], 0, 1)
        feats[:, :3] = np.clip(HueSaturationTranslation.hsv_to_rgb(hsv), 0, 255)

        return coords, feats, labels


class RandomDropout(PointCloudTransform, VoxelTransform):

    def __init__(self, dropout_ratio=0.2):
        self.dropout_ratio = dropout_ratio

    def __call__(self, coords, feats, labels):
        N = len(coords)
        inds = np.random.choice(N, int(N * (1 - self.dropout_ratio)), replace=False)
        return coords[inds], feats[inds], labels[inds]


class RandomScale(PointCloudTransform):

    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __call__(self, coords, feats, labels):
        # coords: N x 3
        scale = np.random.uniform(low=self.low, high=self.high, size=[3])
        coords = np.multiply(coords, scale)
        return coords, feats, labels


class RandomTranslate(PointCloudTransform):
    '''
    For point cloud is going to be voxelized, please use RandomPositiveTranslate
    '''
    def __init__(self, trans_bound):
        '''
            trans_bound is either
                1. Per axis such as ((-5, 5), (0, 0), (-10, 10))
                2. Same for each axis (-0.1, 0.2)
        '''
        self.trans_bound = trans_bound

    def __call__(self, coords, feats, labels):
        # coords: N x 3
        if len(self.trans_bound) == 3:
            for axis in range(3):
                trans = np.random.uniform(self.trans_bound[axis][0], self.trans_bound[axis][0])
                coords[:, axis] += trans
        else:
            trans = np.random.unitform(self.trans_bound[0], self.trans_bound[1], size=[3])
            coords += trans

        return coords, feats, labels


class RandomPositiveTranslate(PointCloudTransform):
    '''
        PositiveTranslate will move the pointcloud to positive space and apply only positive translation.
        This is better for point clouds that are going to be voxelized later.
    '''
    def __init__(self, trans_bound):
        '''
            trans_bound is a list of positive bound for each axis
        '''
        self.trans_bound = trans_bound

    def __call__(self, coords, feats, labels):
        # coords: N x 3
        assert len(self.trans_bound) == 3

        coords = coords - coords.min(0)
        for axis in range(3):
            if self.trans_bound[axis] > 0:
                trans = np.random.uniform(0, self.trans_bound[axis])
                coords[:, axis] += trans
                coords += trans

        return coords, feats, labels


def gen_rotation_matrix(axis, angle):
    if axis == 'x' or axis == 0:
        return np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ], np.float32)
    elif axis == 'y' or axis == 1:
        return np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ], np.float32)
    elif axis == 'z' or axis == 2:
        return np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ], np.float32)
    else:
        return NotImplemented


def apply_rotation(coords, rotation_matrix, around_center):
    if not around_center:
        coords = np.dot(coords, rotation_matrix)
    else:
        # move center to origin and move back after
        coordmax = np.max(coords, axis=0)
        coordmin = np.min(coords, axis=0)
        center = (coordmax + coordmin) / 2
        coords = np.dot(coords - center, rotation_matrix) + center
    return coords


class Random360Rotate(PointCloudTransform):
    ''' Apply on single axis '''
    def __init__(self, axis='z', around_center=True):
        self.axis = axis
        self.around_center = around_center

    def __call__(self, coords, feats, labels):
        angle = np.random.uniform() * 2 * np.pi
        rotation_matrix = gen_rotation_matrix(self.axis, angle)
        coords = apply_rotation(coords, rotation_matrix, self.around_center)
        return coords, feats, labels


class RandomFixAnglesRotate(PointCloudTransform):
    def __init__(self, angles=[0, 90, 180, 270], axis='z', around_center=True):
        self.axis = axis
        self.around_center = around_center
        self.angles = angles

    def __call__(self, coords, feats, labels):
        angle = np.random.choice(self.angles) / 180.0 * np.pi
        rotation_matrix = gen_rotation_matrix(self.axis, angle)
        coords = apply_rotation(coords, rotation_matrix, self.around_center)
        return coords, feats, labels


class RandomRotateEachAxis(PointCloudTransform):
    '''
        Apply on each axis
        rotate_bound: list of random angle bounds
            ((-np.pi / 32, np.pi / 32), (-np.pi / 32, np.pi / 32), (-np.pi, np.pi))
    '''
    def __init__(self, rotate_bound, around_center=True):
        self.rotate_bound = rotate_bound
        self.around_center = around_center

    def __call__(self, coords, feats, labels):
        mats = []
        for axis, (low, high) in enumerate(self.rotate_bound):
            angle = np.random.uniform(low, high)
            rotation_matrix = gen_rotation_matrix(axis, angle)
            mats.append(rotation_matrix)
        np.random.shuffle(mats)
        rotation_matrix = np.eye(3)
        for mat in mats:
            rotation_matrix = rotation_matrix @ mat

        coords = apply_rotation(coords, rotation_matrix, self.around_center)
        return coords, feats, labels


class RandomPerturbationRotate(PointCloudTransform):
    '''
    Small random rotation on each axis
    '''
    def __init__(self, angle_sigma=0.06, angle_clip=0.18, around_center=True):
        self.angle_sigma = angle_sigma
        self.angle_clip = angle_clip
        self.around_center = around_center

    def __call__(self, coords, feats, labels):
        angles = self.angle_sigma * np.random.randn(3)
        angles = np.clip(angles, -self.angle_clip, self.angle_clip)
        rotation_matrix = np.eye(3)
        for angle in angles:
            rotation_matrix = rotation_matrix @ gen_rotation_matrix(self.axis, angle)
        coords = apply_rotation(coords, rotation_matrix, self.around_center)
        return coords, feats, labels


class RandomShuffle(PointCloudTransform):
    def __init__(self):
        return

    def __call__(self, coords, feats, labels):
        N, _ = coords.shape
        perm = np.random.permutation(N)
        return coords[perm, :], feats[perm, :], labels[perm]


class RandomHorizontalFlip(VoxelTransform):

    def __init__(self, upright_axis):
        """
        upright_axis: axis index among x,y,z, i.e. 2 for z
        """
        self.upright_axis = {'x': 0, 'y': 1, 'z': 2}[upright_axis.lower()]
        # Use the rest of axes for flipping.
        self.horz_axes = set(range(3)) - set([self.upright_axis])

    def __call__(self, coords, feats, labels):
        for curr_ax in self.horz_axes:
            if random.random() < 0.5:
                coord_max = np.max(coords[:, curr_ax])
                coords[:, curr_ax] = coord_max - coords[:, curr_ax]
        return coords, feats, labels


class ElasticDistortion(PointCloudTransform):

    def __init__(self, distortion_params):
        self.distortion_params = distortion_params

    def elastic_distortion(self, coords, feats, labels, granularity, magnitude):
        """Apply elastic distortion on sparse coordinate space.

            pointcloud: numpy array of (number of points, at least 3 spatial dims)
            granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
            magnitude: noise multiplier
        """
        blurx = np.ones((3, 1, 1, 1)).astype('float32') / 3
        blury = np.ones((1, 3, 1, 1)).astype('float32') / 3
        blurz = np.ones((1, 1, 3, 1)).astype('float32') / 3
        coords_min = coords.min(0)

        # Create Gaussian noise tensor of the size given by granularity.
        noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
        noise = np.random.randn(*noise_dim, 3).astype(np.float32)

        # Smoothing.
        for _ in range(2):
            noise = scipy.ndimage.filters.convolve(noise, blurx, mode='constant', cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blury, mode='constant', cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blurz, mode='constant', cval=0)

        # Trilinear interpolate noise filters for each spatial dimensions.
        ax = [
                np.linspace(d_min, d_max, d)
                for d_min, d_max, d in zip(coords_min - granularity, coords_min + granularity *
                                           (noise_dim - 2), noise_dim)
        ]
        interp = scipy.interpolate.RegularGridInterpolator(ax, noise, bounds_error=0, fill_value=0)
        coords += interp(coords) * magnitude
        return coords, feats, labels

    def __call__(self, coords, feats, labels):
        for granularity, magnitude in self.distortion_params:
            coords, feats, labels = self.elastic_distortion(coords, feats, labels, granularity, magnitude)
        return coords, feats, labels


class GridSubsampling(PointCloudTransform):
    # TODO
    pass


class Voxelize(PointCloudTransform):
    # TODO
    pass


# Example usage
if __name__ == '__main__':
    ROTATE_AXIS = 'z'

    # Basic point cloud augmentation
    transform1 = Compose([
        RandomApply([
            ElasticDistortion([(0.2, 0.4), (0.8, 1.6)])
        ], 0.95),
        RandomApply([
            RandomTranslate([(-0.2, 0.2), (-0.2, 0.2), (0, 0)])
        ], 0.95),
        Random360Rotate(ROTATE_AXIS, around_center=True),
        RandomApply([
            RandomRotateEachAxis([(-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (0, 0)])
        ], 0.95),
        RandomApply([
            RandomScale(0.9, 1.1)
        ], 0.95),
    ])

    # Feature and voxel transform
    transform2 = Compose([
        RandomApply([RandomDropout(0.2)], 0.5),
        RandomApply([RandomHorizontalFlip(ROTATE_AXIS)], 0.95),
        RandomApply([ChromaticAutoContrast()], 0.2),
        RandomApply([ChromaticTranslation(0.1)], 0.95),
        RandomApply([ChromaticJitter(0.05)], 0.95)
    ])
    for i in range(1000):
        coords = np.random.rand(100, 3) * 10
        feats = np.random.randint(0, 256, size=(100, 3))
        labels = np.random.randint(0, 20, size=100)
        coords, feats, labels = transform1(coords, feats, labels)
        coords, feats, labels = transform2(coords, feats, labels)
    print(coords)
    print(feats)

