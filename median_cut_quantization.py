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
        #  List of colors used in this image.
        for count, pixel in img.getcolors(img.width * img.height):
            self.colors_ += [pixel] * count

        # Create ColorSpace object
        color_subspaces = [ColorSpace(self.colors_)]

        while len(color_subspaces) < self.n_colors:
            # Sort color subspaces wrt to max_range
            color_subspaces.sort()
            color_subspaces += color_subspaces.pop().split()

        # Calculate average of all subspaces
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

