# ColorQuantization

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install necessary libraries.

```bash
pip install -r requirements.txt
```

## Usage

```python
from PIL import Image

img = Image.open('img_name.jpg')
n_colors = 16. # Desired number of colors
mcm = MedianCutQuantization(n_colors)
mcm.fit(img)

```
