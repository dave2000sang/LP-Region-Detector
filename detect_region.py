import cv2
import numpy as np

def resize_img(img, width=500.0):
	cur_width = img.shape[1]
	scale_factor = width / cur_width
	new_width = int(cur_width * scale_factor)
	new_height = int(img.shape[0] * scale_factor)
	return new_width, new_height


GRAYSCALE = 0
img = cv2.imread('test-plate.jpg', GRAYSCALE)

w, h = resize_img(img, 1000.0)
img_resized = cv2.resize(img, dsize=(w, h), interpolation=cv2.INTER_LANCZOS4)

cv2.imwrite('processed2.jpg', img_resized)

img_bilateral = cv2.bilateralFilter(img_resized, 15, 30, 30)
cv2.imwrite('bilateral.jpg', img_bilateral)

img_edges = cv2.Canny(img_bilateral, 100, 200)
cv2.imwrite('edges.jpg', img_edges)

contours, hierarchy = cv2.findContours(img_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# iterate through contours

for contour in contours:
	perimeter = cv2.arcLength(contour, True)
	approx = cv2.approxPolyDP(contour, perimeter * 0.02, True)
	if len(approx) == 4:
		plate_region = approx

		print(plate_region)
		img_copy = img_resized.copy()
		img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2RGB)
		cv2.drawContours(img_copy, [plate_region], -1, (255,0,0), 3)
		cv2.imshow('img with contours', img_copy)
		cv2.waitKey(0)
