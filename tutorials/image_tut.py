import numpy as np
from imageio import imread, imsave
from PIL import Image
from scipy.spatial.distance import pdist, squareform

##  USING IMAGEIO    
img = imread('Karen1.jpg')             # opens image as array
print(img.dtype, img.shape)

img_tinted = img * [1, 0.95, 0.9]

# imsave('Karen1_tinted.jpg', img_tinted)


##  USING PIL
img1 = Image.open('Karen1.jpg')
img1_array = np.array(img1)
print(img1_array.dtype, img1_array.shape)

img1_array_tinted = img1_array * [1, 0.95, 0.9]
img1_tinted = Image.fromarray(np.uint8(img1_array))
img1_tinted = img1_tinted.resize((300,300))
# img1_tinted.show()

##  DISTANCE BETWEEN POINTS
x = np.array([[0,1], [1, 0], [ 2, 0]])
print(x)
d = squareform(pdist(x, 'euclidean'))
print(d)