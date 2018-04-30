import cv2
import time
from datetime import datetime
import pandas
import numpy as np

def diffImg(t0, t1, t2):              # Function to calculate difference between images.
  d1 = cv2.absdiff(t2, t1)
  d2 = cv2.absdiff(t1, t0)
  return cv2.bitwise_and(d1, d2)

first_frame=None
status_list=[None,None]
times=[]
df=pandas.DataFrame(columns=["Start","End"])

threshold = 78000                     # Threshold for triggering "motion detection"
video = cv2.VideoCapture(0)

# Read three images first:
t_minus = cv2.cvtColor(video.read()[1], cv2.COLOR_RGB2GRAY)
t = cv2.cvtColor(video.read()[1], cv2.COLOR_RGB2GRAY)
t_plus = cv2.cvtColor(video.read()[1], cv2.COLOR_RGB2GRAY)
# Lets use a time check so we only take 1 pic per sec
timeCheck = datetime.now().strftime('%Ss')

while True:
    check, frame = video.read()

    totalDiff = cv2.countNonZero(diffImg(t_minus, t, t_plus))	# this is total difference number
    text = "threshold: " + str(totalDiff)			# make a text showing total diff.
    cv2.putText(frame, text, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)   # display it on screen
    
    

    status=0

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(21,21),0) #21,21 is width and height of Gaussian Kernel and 0 is for standard deviation

    if first_frame is None:
        first_frame = gray
        continue

    delta_frame = cv2.absdiff(first_frame,gray)

    thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]#30 is threshold limit. 255(White) is value assigned to values more than 30

    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2) #dilate is for smoothing the areas that area coming as holes


    #Contour Detection
    (_,cnts,_)=cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #RETR_EXTERNAL is to draw the external contours

    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue
        status = 1

        if totalDiff > threshold and timeCheck != datetime.now().strftime('%Ss') and status == 1:
            dimg= video.read()[1]
            cv2.imwrite(datetime.now().strftime('%Y%m%d_%Hh%Mm%Ss%f') + '.jpg', dimg)

        

        (x,y,w,h) = cv2.boundingRect(contour) #drawing a rectangle
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)

    status_list.append(status)

    status_list = status_list[-2:]


    if status_list[-1]==1 and status_list[-2]==0: #status_list[0,1] i.e. when it finds an motion
        times.append(datetime.now())
    if status_list[-1]==0 and status_list[-2]==1: #status_list[1,0] i.e. when the motion ends
        times.append(datetime.now())


    timeCheck = datetime.now().strftime('%Ss')
    # Read next image
    t_minus = t
    t = t_plus
    t_plus = cv2.cvtColor(video.read()[1], cv2.COLOR_RGB2GRAY)
        
    cv2.imshow("Gray Frame",gray)
    cv2.imshow("Delta Frame",delta_frame)
    cv2.imshow("Threshold Frame",thresh_frame)
    cv2.imshow("Color Frame",frame)


    key = cv2.waitKey(1) # 1 is time in ms

    #print(gray)
    #print(delta_frame)

    if key==ord('q'):
        if status==1:
            times.append(datetime.now())
        break
    
print(status_list)
print(times)

for i in range(0,len(times),2):
    df=df.append({"Start":times[i],"End":times[i+1]},ignore_index=True)

df.to_csv("Times.csv")

video.release()

cv2.destroyAllWindows()
