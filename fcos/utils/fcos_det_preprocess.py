import numpy as np
from PIL import Image

def preprocess_(image, input_shape):
    """
    preprocess includes color-converting, rescale, crop, normalize
    image: PIL.Image.Image
    input_shape: (h, w)
    """
    if isinstance(image, str):
        filename = image
        image = Image.open(image)
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    assert (isinstance(image, Image.Image))

    w_ori, h_ori = image.size
    h, w = input_shape
    image = image.convert("RGB")

    if w_ori > h or h_ori > w:
        scale = max(1.0*w_ori / w, 1.0*h_ori / h)
        image.thumbnail((w,h), Image.BILINEAR)
    else:
        scale = 1

    image = image.crop((0, 0, w, h))
    img_data = np.array(image).reshape((1, h, w, 3))
    img_data = img_data / 256. - 0.5

    return img_data, scale, w_ori, h_ori
    
    
