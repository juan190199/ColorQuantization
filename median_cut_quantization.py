import numpy as np
from PIL import Image


class MedianCutQuantization:
    """

    """
    def __init__(self, n_colors):
        """
        Initialize MedianCutQuantization object

        :param n_colors: int
            Number of colors for an image to be quantized to
        """
        self.n_colors = n_colors
        self.colors_ = []

    def fit(self, img):
        """
        Quantize image to a given number of colors

        :param img: PIL Image object
            Image to be quantized

        :return:
        """
        merged = img.copy()
        # List of colors used in this image.
        # Obtain color for all pixels. getcolors returns number of pixels with determined RGB value
        for count, pixel in img.getcolors(img.width * img.height):
            self.colors_ += [pixel] * count

        # Create ColorSpace object
        color_subspaces = [ColorSpace(self.colors_)]

        while len(color_subspaces) < self.n_colors:
            # Sort color subspaces wrt to max_range
            color_subspaces.sort()
            color_subspaces += color_subspaces.pop().split()

        # Calculate average of RGB values of all subspaces
        median_cut_img = [subspace.avg() for subspace in color_subspaces]
        return self.merge_palette(merged, median_cut_img)

    def merge_palette(self, img, palette):
        """
        Create new image by merging original and new color palette

        :param img: PIL image
            Image to be quantized

        :param palette: List
            List with RGB values of all colors in img
        :return:
        """
        # Avoid extra columns of background color by calculating width for each color
        color_width = img.width / len(palette)
        color_height = int(max(100, color_width))
        color_size = (int(color_width), color_height)
        color_x = 0
        color_y = img.height

        # Create a new image to paste the original in and all the colors of the palette
        merged = Image.new('RGB', (img.width, img.height + color_height))
        # Add in the original image
        merged.paste(img)
        # Create a square of each color and insert each, side-by-side, into the palette
        for color in palette:
            color = Image.new('RGB', color_size, color)
            merged.paste(color, (int(color_x), color_y))
            color_x += color_width

        return merged


class ColorSpace(object):
    def __init__(self, colors):
        """
        Initialize ColorSpace object

        :param colors: list
            List of RGB values for all colors in image
        """
        self.colors_ = colors or []
        self.red_ = [r[0] for r in colors]
        self.green_ = [g[1] for g in colors]
        self.blue_ = [b[2] for b in colors]
        self.size = (max(self.red_) - min(self.red_),
                     max(self.green_) - min(self.green_),
                     max(self.blue_) - min(self.blue_))
        self.max_range = max(self.size)

        self.max_dim = self.size.index(self.max_range)

    def avg(self):
        """
        Take average of pixels in a color subspace

        :return: tuple of integers
            Average of pixel values in respective subspace for all dimensions
        """
        r = np.mean(self.red_).astype(np.int64)
        g = np.mean(self.green_).astype(np.int64)
        b = np.mean(self.blue_).astype(np.int64)

        return r, g, b

    def split(self):
        """
        Split color subspace

        :return: tuple of ColorSpace objects
            Color subspaces after splitting
        """
        median = np.median(self.colors_).astype(np.int64)
        # Sort the colors wrt. pixel values of dimension with max range
        colors = sorted(self.colors_, key=lambda c: c[self.max_dim])
        # Split colors
        return ColorSpace(colors[:median]), ColorSpace(colors[median:])

    def __lt__(self, other):
        """
        Comparison color spaces

        :param other:
        :return:
        """
        return self.max_range < other.max_range


