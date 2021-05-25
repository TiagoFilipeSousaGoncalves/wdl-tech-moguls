# Credits: https://towardsdatascience.com/count-number-of-cars-in-less-than-10-lines-of-code-using-python-40208b173554

# Imports
import cv2
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox

# Read image
im = cv2.imread('dset_img_2.jpg')

# Count cars
bbox, label, conf = cv.detect_common_objects(im)
output_image = draw_bbox(im, bbox, label, conf)

# Show
print(int(label.count('car')))
plt.imshow(output_image)
plt.show()
print('Number of cars in the image is '+ str(label.count('car')))