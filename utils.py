import numpy as np

from sympy.utilities.iterables import multiset_combinations


class HistogramQuantization:
    """
    Pixel-wise Vector Quantization of an image, reducing the number of colors required to show the image by splitting
    color space into buckets of the same size
    """

    def __init__(self, buckets_per_dim):
        """

        :param buckets_per_dim: int

        """
        self.max_range_ = 255
        self.buckets_per_dim_ = buckets_per_dim
        self.bucket_size_ = self.max_range_ / self.buckets_per_dim_

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
        self.max_range_ = 255
        self.n_splits_ = n_splits
        self.dim_cut_ = []  # Store cuts per dimension

    def _split_dimension(self, img_arr):
        """

        :param img:
        :param img_arr:
        :param n_splits:
        :return:
        """
        # Calculate dimension with largest range
        dim, min_idx, max_idx = self._max_range(img_arr)

        # Calculate median for dimension with largest range
        median_idx = self._calculate_median_pixel(min_idx, max_idx, dim, img_arr)
        # Append cut
        pix_cut_idx = median_idx.astype(np.int64)
        pix_cut_value = img_arr[pix_cut_idx, dim]
        self.dim_cut_.append((dim, pix_cut_value))

    def _max_range(self, img_arr):
        """
        Calculate start point and end point of maximal range
        :return:
        """
        cuts0 = np.sort([cut[1] for cut in self.dim_cut_ if cut[0] == 0] + [0, self.max_range_])
        cuts1 = np.sort([cut[1] for cut in self.dim_cut_ if cut[0] == 1] + [0, self.max_range_])
        cuts2 = np.sort([cut[1] for cut in self.dim_cut_ if cut[0] == 2] + [0, self.max_range_])

        max = []
        ranges0 = np.diff(cuts0)
        max0 = np.max(ranges0)
        max.append(max0)
        argmax0 = np.argmax(ranges0)
        ranges1 = np.diff(cuts1)
        max1 = np.max(ranges1)
        max.append(max1)
        argmax1 = np.argmax(ranges1)
        ranges2 = np.diff(cuts2)
        max2 = np.max(ranges2)
        max.append(max2)
        argmax2 = np.argmax(ranges2)

        argmax = np.argmax(max)

        if argmax == 0:
            # Dimension with largest range
            dim = 0
            max_cut = cuts0[argmax0 + 1]
            min_cut = cuts0[argmax0]
            if max_cut == self.max_range_:
                argmax = img_arr.shape[0] - 1
            else:
                argmax = np.argmax(img_arr[:, dim] <= max_cut)
            argmin = np.argmin(img_arr[:, dim] >= min_cut)
            idx_range = img_arr[argmin:argmax, dim]
        elif argmax == 1:
            # Dimension with largest range
            dim = 1
            max_cut = cuts1[argmax1 + 1]
            min_cut = cuts1[argmax1]
            if max_cut == self.max_range_:
                argmax = img_arr.shape[0] - 1
            else:
                argmax = np.argmax(img_arr[:, dim] <= max_cut)
            argmin = np.argmin(img_arr[:, dim] > min_cut)
            idx_range = img_arr[argmin:argmax, dim]
        elif argmax == 2:
            # Dimension with largest range
            dim = 2
            max_cut = cuts2[argmax2 + 1]
            min_cut = cuts2[argmax2]
            if max_cut == self.max_range_:
                argmax = img_arr.shape[0] - 1
            else:
                argmax = np.argmax(img_arr[:, dim] <= max_cut)
            argmin = np.argmin(img_arr[:, dim] > min_cut)
            idx_range = img_arr[argmin:argmax, dim]

        return dim, argmin, argmax



    def _calculate_median_pixel(self, sp, ep, dim, img_array):
        """
        :param sp: int
            Start index for the largest range with dimension dim

        :param ep: int
            End index for the largest range with dimension dim

        :param dim: int
            Dimension with max range

        :param img_array: np.ndarray of shape (n_pixels, 3)
            Pixels belonging to specific subspace

        :return: int
            Median of img_array within a given dimension
        """
        # Calculate median index for maximum range
        return np.median(list(range(sp, ep + 1)))


    def fit(self, img, img_arr):
        """
        Reduce number of colors of image by averaging pixels within buckets

        :param img: np.ndarray of shape (h, w, 3)
            Image dimension

        :param img_arr: np.ndarray of shape (n_pixels, 3)
            Pixels belonging to specific subspace

        :return: np.ndarray of shape (n_pixels, 3)
            Quantized image
        """
        self.out = np.copy(img)

        for split in range(self.n_splits_):
            self._split_dimension(img_arr)

        # Calculate average of all subspaces
        self._calculate_avg_pixels(img_arr)


    def _calculate_avg_pixels(self, img_arr):
        """
        Calculate average pixel for all subspaces

        :para img: np.ndarray of shape (h, w, 3)
            Image dimensions

        :param img_arr: np.ndarray of shape (n_pixels, 3)
            Pixels belonging to specific subspace

        :return: np.ndarray of shape (n_subspaces, 3)
            Average pixel for all subspaces
        """
        medians0 = np.sort([cut[1] for cut in self.dim_medians_ if cut[0] == 0] + [0, self.max_range_])
        medians1 = np.sort([cut[1] for cut in self.dim_medians_ if cut[0] == 1] + [0, self.max_range_])
        medians2 = np.sort([cut[1] for cut in self.dim_medians_ if cut[0] == 2] + [0, self.max_range_])
        # Determine ranges
        ranges0 = np.diff(medians0)
        ranges1 = np.diff(medians1)
        ranges2 = np.diff(medians2)

        i = j = k = 1

        print(img_arr[i:i + 1, 0])

        # Calculate subspaces
        for i in medians0:
            for j in medians1:
                for k in medians2:
                    np.mean(img_arr[i:i + 1, 0][j:j + 1, 1][k:k+1, 2])
