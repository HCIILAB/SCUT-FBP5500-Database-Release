### convert 'pts' to 'txt' format
import os
import re
import struct
import shutil
import cv2

def get_files(dname, suffix):
	pts_list = []
	for fname in os.listdir(dname):
		if fname.endswith(suffix):
			pts_list += [fname]
	return pts_list

def pts2txt(din, dout, src):
	src_p = os.path.join(din, src)
	data = open(src_p, 'rb').read()
	if len(data) < 692:
		return 0
	points = struct.unpack('i172f', data)
	# print points
	
	dst = src
	dst = dst.replace('pts', 'txt')
	dst_p = os.path.join(dout, dst)
	#print dst_p

	fout = open(dst_p, 'w')
	pnum = len(points[1:])
	for i in range(1, pnum, 2):
		fout.write('%f ' % points[i])
		fout.write('%f\n' % points[i+1])
	fout.close()

	return 1

def display(img_dir, dst, new_root):
	img_name = os.path.basename(img_dir)
	new_img_dir = os.path.join(new_root, img_name)
	txt_name = img_name.replace('jpg', 'txt')
	txt_dir = os.path.join(dst, txt_name)
	if not os.path.isfile(txt_dir):
		return 0
	txt = open(txt_dir, 'r')
	lines = txt.readlines()
	img = cv2.imread(img_dir)
	for i in range(len(lines)):
		radius = 3
		color = (255, 55, 55)
		x = int(float(lines[i].split(' ')[0]))
		y = int(float(lines[i].split(' ')[1]))
		cv2.circle(img, (x, y), radius, color, -1)
	cv2.imshow('img', img)
	cv2.waitKey(0)
	cv2.imwrite(new_img_dir, img)

	return 1

	
def main():
	src = './facial landmark'
	dst = './landmark_txt'

	if not os.path.exists(dst):
		os.mkdir(dst)

	pts_list = get_files(src, 'pts')	
	for pts in pts_list:
		flag = pts2txt(src, dst, pts)


	jpg_root = './Images'
	new_root = './Save'
	os.mkdir(new_root) if not os.path.exists(new_root) else None
	jpg_list = map(lambda f: os.path.join(jpg_root, f), os.listdir(jpg_root))
	for jpg in jpg_list:
		flag = display(jpg, dst, new_root)


if __name__ == '__main__':
	main()



