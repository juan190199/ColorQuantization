# ColorQuantization

Conceptually, extracting a palette from an image is simple – the goal is to detect a set of colors that capture the mood of the image. How large or small this set of colors should be can be subjective. All of the following approaches allow some control over the number of colors to extract. 

The first step in color palette extraction is to represent the colors of an image in a mathematical way. Since each pixel’s color can be represented in RGB format, we can plot each pixel in a three-dimensional RGB (red, green, blue) color space.

## Histogram Approach

This can be done by splitting the three-dimensional RGB space into a uniform grid and taking into account the pixels in each color subspace. The average pixel values in these color subspaces give us the resulting palette.

## Median Cut Algorithm

Median cut is a nice extension of the simple histogram approach. Instead of creating a fixed histogram grid, median cut partitions the space in a more intelligent way.

The algorithm is described as follow:
1. Find out which color channel (red, green, or blue) among the pixels in the color space has the greatest range, and sort the pixels according to that channel's values.
2. After the color space has been sorted, move the upper half of the pixels into a new bucket (median cut)
3. Repeat the process on both buckets and successive buckets
4. Average the pixels in each color subspace


## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install necessary libraries.

```bash
pip install -r requirements.txt
```
