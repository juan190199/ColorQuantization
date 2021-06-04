import numpy as np

from sympy.utilities.iterables import multiset_combinations


class HistogramPaletteBuilder:
    """
    Pixel-wise Vector Quantization of an image, reducing the number of colors required to show the image by splitting
    color space into buckets of the same size
    """

    def __init__(self, buckets_per_dim):
        """

        :param buckets_per_dim: int

        """
        MAX_DIM = 255
        self.buckets_per_dim_ = buckets_per_dim
        self.bucket_size_ = MAX_DIM / self.buckets_per_dim_

    def _get_buckets(self, img_arr):
        """
        Identify bucket for each pixel

        :param img_arr: np.ndarray of shape (w * h, 3)
            Array of pixels for each of the RGB dimensions

        :return: np.ndarray of shape (img_arr.shape[0],)
            Array with index of buckets for each pixel
        """
        pixels_dim_buckets = np.floor(img_arr / self.bucket_size_).astype(np.int64)
        return pixels_dim_buckets

    def fit(self, img_arr):
        """
        Reduce number of colors of image by averaging pixels within buckets

        :param img_arr: np.ndarray of shape (n_pixels, 3)
            Image reshaped into 2D array with 3 dimensions for RGB channels

        :return: np.ndarray of shape (n_pixels, 3)
            Quantized image
        """
        self._n_pixels = img_arr.shape[0]
        # Initialize quantized image array
        quantized_img_arr = np.empty(shape=img_arr.shape)
        # Identify bucket for each pixel
        pixels_dim_buckets = self._get_buckets(img_arr)
        # Calculate average of each bucket
        avg_pixels, bucket_mask = self._calculate_avg_pixels(pixels_dim_buckets)
        # Set pixels to respective average
        for bucket in range(len(bucket_mask)):
            quantized_img_arr[bucket_mask[bucket]] = avg_pixels[bucket_mask[bucket]]

        return img_arr

    def _calculate_avg_pixels(self, pixels_dim_buckets):
        """
        Calculate average pixel for each bucket

        :param pixels_dim_buckets: np.ndarray of shape (img_arr.shape[0], 3)
            Pixel buckets per dimension. img_arr.shape[0] corresponds to the number of pixels

        :return: np.ndarray of shape (buckets_per_dim, 3), array of len n_buckets
            Average pixel per bucket and pixels idx in respective bucket
        """
        # Calculate all possible buckets
        arr_buckets_per_dim = np.array(range(self.buckets_per_dim_))
        buckets = np.array(np.meshgrid(arr_buckets_per_dim, arr_buckets_per_dim, arr_buckets_per_dim)).T.reshape(-1, 3)
        # Indexing buckets to store average per bucket
        avg_arr_idx_buckets = np.zeros(shape=(buckets.shape[0], 3))
        # Pixel idx in respective bucket
        bucket_mask = []

        # Unique buckets
        u_buckets = np.unique(pixels_dim_buckets, axis=0)

        # Find pixels per bucket and take average
        for idx in range(buckets.shape[0]):
            if buckets[idx] in u_buckets:
                # Get all pixels in bucket idx
                bool_pixel_arg = np.array([buckets[idx] == pixels_dim_buckets])
                mask = np.prod(bool_pixel_arg, axis=0).prod(axis=1)

                # Calculate average of the bucket idx
                avg = np.mean(pixels_dim_buckets[mask], axis=0)
                avg_arr_idx_buckets[idx] = avg
                bucket_mask.append(mask)
            else:
                avg_arr_idx_buckets[idx] = 0.0

        return avg_arr_idx_buckets, bucket_mask


class MedianCutPaletteBuilder:
    def __init__(self, n_splits):
        """

        :param n_splits: int

        """
        self.colors = n_splits

    def _split_dimension(self, img, img_arr, n_splits):
        """

        :param img:
        :param img_arr:
        :param n_splits:
        :return:
        """
        if n_splits == 0:
            self._calculate_avg_pixels(img, img_arr)
            return

        r_range = np.max(img_arr[:, 0]) - np.min(img_arr[:, 0])
        g_range = np.max(img_arr[:, 1]) - np.min(img_arr[:, 1])
        b_range = np.max(img_arr[:, 2]) - np.min(img_arr[:, 2])

        # Range comparison
        max_range_dim = 0
        if r_range >= g_range and r_range >= b_range:
            max_range_dim = r_range
            dim = 0
        elif g_range >= r_range and g_range >= b_range:
            max_range_dim = g_range
            dim = 1
        elif b_range >= r_range and b_range >= g_range:
            max_range_dim = b_range
            dim = 2

        # Compute median pixel value for max range dimension
        median_pixel = self._calculate_median(img_arr, dim)

        # Determine left and right subspace
        l_mask = img_arr[:, dim] < median_pixel
        l_subspace_img_arr = img_arr[l_mask]
        r_mask = img_arr[:, dim] >= median_pixel
        r_subspace_img_arr = img_arr[~l_mask]

        assert img_arr.shape[0] == l_subspace_img_arr.shape[0] + r_subspace_img_arr.shape[0]

        # Split pixels into subspaces
        self._split_dimension(img, l_subspace_img_arr, n_splits - 1)
        self._split_dimension(img, r_subspace_img_arr, n_splits - 1)

    def _calculate_median(self, img_array, dim):
        """

        :param img_array: np.ndarray of shape (n_pixels, 3)
            Pixels belonging to specific subspace

        :param dim: int
            Dimension with max range

        :return: int
            Median of img_array within a given dimension
        """
        return np.median(img_array[:, dim]).astype(np.int64)

    def fit(self, img, img_arr):
        """
        Reduce number of colors of image by averaging pixels within buckets

        :param img: np.ndarray of shape (h, w, 3)
            Image dimension

        :param img_arr: np.ndarray of shape (n_pixels, 3)
            Image reshaped into 2D array with 3 dimensions for RGB channels

        :return: np.ndarray of shape (n_pixels, 3)
            Quantized image
        """
        self.out = np.copy(img)
        return self._split_dimension(img, img_arr, self.colors)

    def _calculate_avg_pixels(self, img, img_arr):
        """
        Calculate average pixel for each bucket

        :para img: np.ndarray of shape (h, w, 3)
            Image dimension

        :param img_arr: np.ndarray of shape (n_pixels, 3)
            Image reshaped into 2D array with 3 dimensions for RGB channels

        :return: np.ndarray of shape (buckets_per_dim, 3), array of len n_buckets
            Average pixel per bucket and pixels idx in respective bucket
        """
        r_avg = np.mean(img_arr[:, 0])
        g_avg = np.mean(img_arr[:, 1])
        b_avg = np.mean(img_arr[:, 2])

        for data in img_arr:
            self.out[data[3]][data[4]] = [r_avg, g_avg, b_avg]

