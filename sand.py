import cv2
#import numpy 


palm_model = cv2.CascadeClassifier('/home/avdesh/AditiEdge/data/cascade.xml')

list_path='/home/avdesh/AditiEdge/pos/pos.txt'
with open(list_path) as mynewfile:
	contents=mynewfile.read().splitlines()
	for a in contents:
		image = cv2.imread('/home/avdesh/AditiEdge/pos/'+a+'.jpg',0)
		#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		scale_factor = 1.1
		min_neighbors = 1
		min_size = (20, 20)


		palms = palm_model.detectMultiScale(image, scaleFactor = scale_factor, minNeighbors = min_neighbors, minSize = min_size)
		for (x,y,w,h) in palms:
			gray = cv2.rectangle(image,(x,y),(x+w,y+h),(50,50,200),2)
			f=open('/home/avdesh/AditiEdge/pos/predicted.txt','a')
			f.write('%i %i %i %i\n'%(x,y,w,h))
f.close()

#cam = cv2.VideoCapture(0)





