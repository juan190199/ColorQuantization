import matplotlib.pyplot as plt


def viz_image(img, title, save=False, img_name=None):
    plt.clf()
    plt.axis('off')
    plt.title(title)
    if save is True:
        plt.imsave('images/quantized_images/' + img_name + '.jpg', img)
    plt.imshow(img)
