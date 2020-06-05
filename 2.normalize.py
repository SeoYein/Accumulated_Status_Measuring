import os
from utils import *
from PIL import Image
from PIL import ImageStat
import operator

root_dir = "D:\\(0806)new_data\\roi\\roi_0\\all_dataset_roi_train"
# source_dir = 'all_dataset_mod_resize_crop_90'

mean = [0, 0, 0]
std = [0, 0, 0]

cnt = 0

for root, d_names, f_names in os.walk(root_dir):
	for filename in f_names:
		ext = os.path.splitext(filename)[-1]
		if ext == '.png' or ext == '.jpg':
			img = Image.open(os.path.join(root, filename))

		if root.find('Train'):
			stat = ImageStat.Stat(img)

			mean[0] = mean[0] + stat.mean[0]
			mean[1] = mean[1] + stat.mean[1]
			mean[2] = mean[2] + stat.mean[2]

			std[0] = std[0] + stat.stddev[0]
			std[1] = std[1] + stat.stddev[1]
			std[2] = std[2] + stat.stddev[2]

			if cnt % 100 == 0:
				print(cnt)
				print(mean)
				print(std)

			cnt = cnt + 1

mean[0] = mean[0] / cnt / 255.0
mean[1] = mean[1] / cnt / 255.0
mean[2] = mean[2] / cnt / 255.0
std[0] = std[0] / cnt / 255.0
std[1] = std[1] / cnt / 255.0
std[2] = std[2] / cnt / 255.0

print(cnt, mean, std)