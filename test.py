import seetaface_api as sf
import os
import cv2

def draw_keypoints(img_path):
	#initialize
	seetaface = sf.SeetaFace()

	#detect faces and landmarks
	faces = seetaface.face_detect(img_path)
	landmarks = seetaface.face_align(img_path)

	#read img
	img = cv2.imread(img_path)

	#draw face rects
	for i in faces:
		img[i.y, i.x:i.x + i.width] = [255, 0, 0]
		img[i.y + i.height, i.x:i.x + i.width] = [255, 0, 0]
		img[i.y:i.y + i.height, i.x] = [255, 0, 0]
		img[i.y:i.y + i.height, i.x + i.width] = [255, 0, 0]

	#draw landmarks
	for i in landmarks:
		for j in range(5):
		    img[i.y[j]-1:i.y[j]+1, i.x[j] - 1: i.x[j]+1] = [255, 0, 0]

	#mkdir
	if not os.path.exists("./Sample_Result"):
		os.mkdir("Sample_Result")

	#save
	cv2.imwrite('Sample_Result/' + img_path.split("/")[-1], img)

def face_verify(img_path1, img_path2):
	seetaface = sf.SeetaFace()
	return seetaface.face_verify(img_path1, img_path2)

draw_keypoints('Sample/Aaron_Peirsol_0001.jpg')
draw_keypoints('Sample/Aaron_Peirsol_0004.jpg')
print face_verify('Sample/Aaron_Peirsol_0001.jpg', 'Sample/Aaron_Peirsol_0004.jpg')
