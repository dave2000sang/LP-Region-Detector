import cv2
import numpy as np

def resize_img(img, width=500.0):
	cur_width = img.shape[1]
	scale_factor = width / cur_width
	new_width = int(cur_width * scale_factor)
	new_height = int(img.shape[0] * scale_factor)
	return new_width, new_height


IMG_WIDTH = 1300.0
GRAYSCALE = 0
img = cv2.imread('test3.jpg', GRAYSCALE)

w, h = resize_img(img, IMG_WIDTH)
img_resized = cv2.resize(img, dsize=(w, h), interpolation=cv2.INTER_LANCZOS4)

# cv2.imwrite('processed2.jpg', img_resized)

img_bilateral = cv2.bilateralFilter(img_resized, 15, 30, 30)
# cv2.imwrite('bilateral.jpg', img_bilateral)

img_edges = cv2.Canny(img_bilateral, 100, 200)
# cv2.imwrite('edges.jpg', img_edges)

contours, hierarchy = cv2.findContours(img_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# iterate through contours

possible_plates = []
rectangles = []

for contour in contours:
	perimeter = cv2.arcLength(contour, True)
	# approx = contour
	approx = cv2.approxPolyDP(contour, perimeter * 0.02, True)
	if len(approx) >= 4 and perimeter > 100:
		plate_region = approx
		min_rect = cv2.minAreaRect(plate_region)
		angle = min_rect[2]
		# print(min_rect)
		width_to_height_ratio = min_rect[1][0]/min_rect[1][1]
		area = min_rect[1][0] * min_rect[1][1]

		if abs(angle) < 2 and width_to_height_ratio > 1.8 and width_to_height_ratio < 4 and area > (IMG_WIDTH/10)**2:
			print(min_rect)
			box = np.int0(cv2.boxPoints(min_rect))
			possible_plates.append(plate_region)
			rectangles.append(box)

print(len(possible_plates))
img_copy = img_resized.copy()
img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2RGB)
cv2.drawContours(img_copy, possible_plates, -1, (0,0,255), 2)
cv2.drawContours(img_copy, rectangles, -1, (0,255,0), 3)
cv2.imshow('img with contours', img_copy)
cv2.waitKey(0)
cv2.imwrite('detected_plate3.jpg', img_copy)
