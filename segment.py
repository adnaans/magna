import numpy as np
import cv2
from matplotlib import pyplot as plt
img = cv2.imread('res/1.jpg')
realimg = cv2.imread('imgs/60002.jpg')

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
groupnum = 1

def floodfill(x, y, groupnum):
    if thresh[x, y] == 0:
        thresh[x, y] = 50 * groupnum
        print("%d %d %g" % (x, y, groupnum))
        if x > 0:
            floodfill(x-1, y, groupnum)
        if x < 15:
            floodfill(x+1, y, groupnum)
        if y > 0:
            floodfill(x, y-1, groupnum)
        if y < 15:
            floodfill(x, y+1, groupnum)

for x in range(16):
    for y in range(16):
        if(thresh[x,y] == 0):
            floodfill(x, y, groupnum)
            minx = -1
            miny = -1
            maxx = -1
            maxy = -1
            for x in range(16):
                for y in range(16):
                    if thresh[x,y] == groupnum * 50:
                        if minx == -1 or x < minx:
                            minx = x
                        if miny == -1 or y < miny:
                            miny = y
                        if maxx == -1 or x > maxx:
                            maxx = x
                        if maxy == -1 or y > maxy:
                            maxy = y
            cv2.rectangle(realimg,(minx,miny),(maxx,maxy),(0,255,0),2)
            groupnum += 1



fig = plt.figure()
plt.imshow(realimg, cmap="Greys")
plt.show()
