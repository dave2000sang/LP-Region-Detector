import cv2

def resize_img(img, width=500.0):
	cur_width = img.shape[1]
	scale_factor = width / cur_width
	new_width = int(cur_width * scale_factor)
	new_height = int(img.shape[0] * scale_factor)
	return new_width, new_height


GRAYSCALE = 0
img = cv2.imread('test-plate.jpg', GRAYSCALE)

w, h = resize_img(img, 500.0)
img_resized = cv2.resize(img, dsize=(w, h), interpolation=cv2.INTER_LANCZOS4)

cv2.imwrite('processed2.jpg', img_resized)
print (img_resized.shape)
