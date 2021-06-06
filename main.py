import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from median_cut_quantization import MedianCutQuantization
from viz import viz_image


def main():
    image_name = 'northern_lights'
    img = Image.open('images/' + image_name + '.jpg')

    n_colors = 16

    MCM = MedianCutQuantization(n_colors=n_colors)
    mcm_quantized_img = MCM.fit(img)

    viz_image(mcm_quantized_img, 'Quantized image: median cut algorithm', save=True, img_name='mcm_quantized_' + image_name)


if __name__ == '__main__':
    main()
