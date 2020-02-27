from __future__ import division
import dlib
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt 
#from mpl_toolkits.mplot3d import Axes3D
import time

frame_width = 640
frame_height = 480
brightness = 150
contrast = 40
exposure = -7

refRealWidth = 2.85	# Using a ball (inches)
eyeRealWidth = 2.656	# Using the distance between Evan's eyes (inches)
focal = 734.94	# Based on Evan's eye and face distance measurements

# For blue HSV mask Best result so far l=(26,80,70), u=(35,255,255)
lower = np.array([26,80,70]) 
upper = np.array([35,255,255])

def pixelToCameraTransform( PCx, PCy, d, f ):
	CCx = int(d * PCx / f)
	CCy = int(d * PCy / f)
	CCz = int(d)

	camCoord = ( CCx, CCy, CCz )
	return camCoord;

##### Parse arguments for individualized facial measurements #####
#parser = argparse.ArgumentParser(description='Calibrate with facial measurements.')
#parser.add_argument('integers', metavar='N', type=int, nargs='+', help='an integer for the accumulator')
#parser.add_argument('--sum', dest='accumulate', action='store_const', const=sum, default=max, help='sum the integers (default: find the max)')


##### Begin Calibration Step. Find the PC of the reference point. 	#####
##### Input: userEyeDist, faceDist					#####
#userEyeDist = input("To calibrate, enter the distance between your eyes in inches (as precise as you'd like): ")
#faceDist such that their eyes go from one end to the other end of the screen
print("Reference calibration step.\nMake sure the reference is in the frame to start video")
time.sleep(.00001)
#input("Press Enter to start")

### Setting the Camera.         Default values are: Exposure = -4.0,    ###
### Brightness = 150.0, Contrast = 40.0, Saturation = 32.0, Hue = 13.0  ###
video = cv2.VideoCapture(1) # 0 = integrated webcam; 1 = USB webcam
video.set(cv2.CAP_PROP_FRAME_WIDTH,frame_width)
video.set(cv2.CAP_PROP_FRAME_HEIGHT,frame_height)
#video.set(cv2.CAP_PROP_BRIGHTNESS,130)

#cam_exposure = video.get(cv2.CAP_PROP_EXPOSURE)
#cam_bright = video.get(cv2.CAP_PROP_BRIGHTNESS)
#cam_contrast = video.get(cv2.CAP_PROP_CONTRAST)
#cam_saturation = video.get(cv2.CAP_PROP_SATURATION)
#cam_hue = video.get(cv2.CAP_PROP_HUE)
#print("Exposure =",cam_exposure,"\nBrightness =",cam_bright,"\nContrast =",cam_contrast,"\nSaturation =",cam_saturation,"\nHue =",cam_hue)

#video.set(cv2.CAP_PROP_EXPOSURE,exposure)
#video.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
#video.set(cv2.CAP_PROP_CONTRAST, contrast)

### Add distortion calibration at this step 	###

### Using contours to find reference (for now) 	###

contours = []
#while (len(contours) == 0):
while True:
    ret, frame = video.read()
    #cv2.imshow("Frame",frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    frameMask = cv2.bitwise_and(frame, frame, mask = mask)
    #filtered = cv2.medianBlur(frameMask,15) 	# Check other filtering techiques (might need better edges)
    
    biFiltered = cv2.bilateralFilter(frameMask,5,200,200)    # where res = source, 4 = diameter of pixel neighborhood, 75 = color space range to filter, 75 = coordinate space range
    filtered = cv2.medianBlur(biFiltered,11) 	# Check other filtering techiques (might need better edges)
    #gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)	 # findContours is a single channel 8-bit so need cvt2gray
    canny = cv2.Canny(filtered, 225, 250)

    (im2, contours, heirarchy) = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)	# findContours modifies the img past to it
            # Need to grab the coordinates after it finds the contours around the colored object

    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:3] # Sort top three contours with biggest area
#	refContour = contours[0] 	# The contour is a list of points each with x, y member variables 
    leftMost = 720
    rightMost = 0
    topMost = 0
    botMost = 640
    if len(contours) == 0:
        continue
    # Find the center of the outer most points
    for i in range(0,len(contours[0])):	    # Look through the first (biggest) contours list of points (i = index)
        leftMost = contours[0][i][0][0] if contours[0][i][0][0] < leftMost else leftMost
        rightMost = contours[0][i][0][0] if contours[0][i][0][0] > rightMost else rightMost
        topMost = contours[0][i][0][1] if contours[0][i][0][1] > topMost else topMost
        botMost = contours[0][i][0][1] if contours[0][i][0][1] < botMost else botMost
    refPixWidth = int(max((rightMost - leftMost),(botMost - topMost)))
 #  print("rightMost=",rightMost,", leftMost=",leftMost,", topMost=",topMost,", botMost",botMost)

#   print("refPixWidth =",refPixWidth)

    if refPixWidth == 0 or refPixWidth <= refRealWidth:
        continue
    #print("leftMost",leftMost,"rightMost",rightMost,"botMost",botMost,"topMost",topMost)
    refPCx = int(np.mean([leftMost, rightMost]))
    refPCy = int(np.mean([topMost, botMost]))

#    print("leftMost",leftMost,"rightMost",rightMost,"botMost",botMost,"topMost",topMost)

#    print("Reference PC right after",(refPCx,refPCy))
    
    cv2.circle(frame,(refPCx,refPCy),10,(0,0,255),-1)
    cv2.putText(frame, "Press enter when the circle is centered.", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),2,cv2.LINE_AA)
    cv2.imshow("Frame",im2)
    #cv2.imshow("Frame",frame)
    #cv2.imshow("Canny",canny)
    #cv2.imshow("Canny2",canny2)
    #cv2.imshow("Gray",gray)
    #cv2.imshow("Filtered",filtered)
    #cv2.imshow("BiFilter",biFiltered)
    
    if cv2.waitKey(80) == 13 :
        break
	
#video.release()
#cv2.destroyAllWindows() 
refDist = focal * refRealWidth / refPixWidth # !!! Needs to be improved to account for the off center positioning
##### End Calibration Step.		#####
##### Output: refPCx, refPCy, refDist  	#####


                # ##### HAAR Cascade for Facial Detection #####
                # ##### Inputs: None 		    	#####
                # face_cascade = cv2.CascadeClassifier('C:\opencv\data\haarcascades\haarcascade_frontalface_default.xml')
                # mouth_cascade = cv2.CascadeClassifier('C:\opencv\data\haarcascades\haarcascade_mouth.xml')
                # eye_cascade = cv2.CascadeClassifier('C:\opencv\data\haarcascades\haarcascade_eye.xml')

                # video = cv2.VideoCapture(1) # 0 = integrated webcam; 1 = USB webcam
                # video.set(cv2.CAP_PROP_FRAME_WIDTH,frame_width)
                # video.set(cv2.CAP_PROP_FRAME_HEIGHT,frame_height)
                # video.set(cv2.CAP_PROP_EXPOSURE,exposure)
                # #video.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
                # #video.set(cv2.CAP_PROP_CONTRAST, contrast)


                # eyes = []
                # mouth = []
                # #while ( len(eyes) != 2 and len(mouth) != 1):
                # while True:
                #     #cv2.destroyAllWindows() 
                #     ret, frame = video.read()
                #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                #     faces = face_cascade.detectMultiScale(gray, 1.3, 4)
                #     cv2.putText(frame, "Distance from mouth to reference = Error.\nDetecting Face...", (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),2,cv2.LINE_AA)
                #     cv2.imshow("Frame",frame)

                #     if len(faces) != 1:
                #         if cv2.waitKey(30) == 13 :  # Needs a wait key or loops too fast to cv2.imshow
                #             break
                #         continue
                    
                #     #print("face detected")
                #     (x,y,w,h) = faces[0]
                #     (fx,fy,fw,fh) = (int(x),int(y),int(w),int(h))
                #     cv2.rectangle(frame, (fx,fy), (fx+fw, fy+fh), (255,0,0), 2)

                #     roi_gray_lower = gray[fy+int(fh/2):fy+fh, fx:fx+fw]	# Labels certain regions of the picture as a variable (Thus it becomes an image object)
                #     roi_gray_upper = gray[fy:fy+int(fh/2), fx:fx+fw]
                #     roi_color_lower = frame[fy+int(fh/2):fy+fh, fx:fx+fw]
                #     roi_color_upper = frame[fy:fy+int(fh/2), fx:fx+fw]

                #     mouth = mouth_cascade.detectMultiScale(roi_gray_lower, 3, 1)
                #     eyes = eye_cascade.detectMultiScale(roi_gray_upper, 1.3, 10)

                #     if( len(eyes) != 2 or len(mouth) != 1):
                #         if cv2.waitKey(30) == 13 :  # Needs a wait key or loops too fast to cv2.imshow
                #             break
                #         continue
                    
                #     #print("Number of eyes", len(eyes))
                #     (mx,my,mw,mh) = mouth[0]  #May not need for loop since there should only be one
                #     cv2.rectangle(roi_color_lower, (mx, my), (mx+mw, my+mh), (0,255,255), 2)        
                #     for (ex,ey,ew,eh) in eyes:
                #         cv2.rectangle(roi_color_upper, (ex, ey), (ex+ew, ey+eh), (255,255,0), 2)

                #     mouPCx = (fx+mx) + mw/2
                #     mouPCy = (fy+my) + mh/2

                #     #cv2.putText(frame, str((centerMX, centerMY)), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),2,cv2.LINE_AA)

                #     eyePixWidth = abs(eyes[0][0] - eyes[1][0])
                #     mouDist = focal * eyeRealWidth / eyePixWidth
                #### End of HAAR Cascade 		#####
                #### Outputs: mouPCx, mouPCy, mouDist 	#####


##### dlib Face Detection   #####
##### Inputs: None          #####
PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat' 
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

#video = cv2.VideoCapture(1)

while True:
    face = []
    while len(face) < 1 :
        ret, frame = video.read() 

        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_small = cv2.resize(img_gray, (0,0), fx=0.75, fy=0.75) # downsize image for more effiecient processing

        face = detector(img_small, 0) # use default dlib detector to find faces in frame
        
    #video.release()
    #cv2.imshow("Frame",img_small)   

#if len(face) >= 1: # only if one face is detected
    shape = predictor(img_small, face[0])
    
    #shape_np = toArray(shape)
    shape_np = np.zeros((68,2), dtype=int) # initialize blank np array for 68 features
    for i in range(0,68):
        shape_np[i] = (shape.part(i).x, shape.part(i).y) # insert the x and y coords of the shape into the np array
    
    #cv2.imshow("Frame",frame)
    #time.sleep(0.00001)
    
    # index 36 - right corner of right eye
    # index 42 - right corner of left eye
    eyedistance = math.sqrt(math.pow((4/3)*shape_np[36][0] - (4/3)*shape_np[42][0],2) + math.pow((4/3)*shape_np[36][1] - (4/3)*shape_np[42][1],2))
    mouDist = round(((focal * eyeRealWidth) / eyedistance),3)
    # using perspective geometry to find mouth distance from eye distance
    
    mouPCx = int((shape_np[51][0] + shape_np[57][0])*(2/3))
    mouPCy = int((shape_np[51][1] + shape_np[57][1])*(2/3))
    # indeces 51 and 57 are the top and bottom of the middle of the mouth, respectively
    
    lipDist1 = math.sqrt(math.pow((4/3)*shape_np[61][0] - (4/3)*shape_np[67][0],2) + math.pow((4/3)*shape_np[61][1] - (4/3)*shape_np[67][1],2))
    lipDist2 = math.sqrt(math.pow((4/3)*shape_np[62][0] - (4/3)*shape_np[66][0],2) + math.pow((4/3)*shape_np[62][1] - (4/3)*shape_np[66][1],2))
    lipDist3 = math.sqrt(math.pow((4/3)*shape_np[63][0] - (4/3)*shape_np[65][0],2) + math.pow((4/3)*shape_np[63][1] - (4/3)*shape_np[65][1],2))
    lipDistValid = ((lipDist1 + lipDist2 + lipDist3)/3)/eyedistance > 0.15
    # using the six different landmarks for the inner middle of the mouth to find the distance between lips
    # 0.10 - 0.02: inverse range of distance within reasonable operating range of robot

    # labeling the landmarks on the screen
    for i, (x,y) in enumerate(shape_np):
    #cv2.putText(img, str(i), (int(x*(4/3)), int(y*(4/3))), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0,255,0), 1,cv2.LINE_AA)
        cv2.circle(frame, (int(x*(4/3)), int(y*(4/3))), 1, (255,255,255), -1)

    cv2.imshow("Frame",frame)
                
##### End dlib Face Detection           #####
##### Outputs: mouPCx, mouPCy, mouDist  #####


##### Reconstruction algorithm to go from pixel to camera coordinates 	#####
##### Inputs: mouPC, mouDist, refPC, refDist 				#####
    #print("refPixWidth",refPixWidth)
    #print("refDist",refDist)

 #   print("leftMost",leftMost,"rightMost",rightMost,"botMost",botMost,"topMost",topMost)

 #   print("Reference PC at",(refPCx,refPCy),"\nmouth at",(mouPCx,mouPCy))
    
    
    ### Shift the x and y pixel coordinates to the center of the screen     ###
    mouPCx_sft = mouPCx - int(frame_width/2)
    mouPCy_sft = (mouPCy - int(frame_height/2)) * -1
    refPCx_sft = refPCx - int(frame_width/2)
    refPCy_sft = (refPCy - int(frame_height/2)) * -1

    refCC = pixelToCameraTransform( refPCx_sft, refPCy_sft, refDist, focal )
    mouCC = pixelToCameraTransform( mouPCx_sft, mouPCy_sft, mouDist, focal )
    print("Reference at",refCC,"\nand mouth at",mouCC)

    refCCMat = np.array([refCC[0],refCC[1],refCC[2]])
    mouCCMat = np.array([mouCC[0],mouCC[1],mouCC[2]])
    mouRefDist = plt.mlab.dist(refCCMat,mouCCMat)
    
    cv2.putText(frame, "Distance from mouth to reference = " + str(mouRefDist), (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),2,cv2.LINE_AA)
    cv2.imshow("Frame",frame)    
##### Outputs: refCC, mouCC 	#####


##### Plot the 3D Reconstruction 	#####
##### Inputs: mouCC, refCC		#####
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    if plt.fignum_exists(1):
        plt.clf()
        ax = fig.add_subplot(111, projection='3d')
        
        x1 = 1 * np.outer(np.cos(u), np.sin(v))
        y1 = 1 * np.outer(np.sin(u), np.sin(v))
        z1 = 1 * np.outer(np.ones(np.size(u)), np.cos(v))

        ax.plot_surface(x1, y1, z1,  rstride=4, cstride=4, color='k', linewidth=0, alpha=0.5)

        x = 1 * np.outer(np.cos(u), np.sin(v)) + refCC[0]
        y = 1 * np.outer(np.sin(u), np.sin(v)) + refCC[2]               # Since we're plotting the y on the z-axis and vice versa
        z = 1 * np.outer(np.ones(np.size(u)), np.cos(v)) + refCC[1]     # Since we're plotting the y on the z-axis and vice versa

        ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='b', linewidth=0, alpha=0.5)

        mx = 1 * np.outer(np.cos(u), np.sin(v)) + mouCC[0]
        my = 1 * np.outer(np.sin(u), np.sin(v)) + mouCC[2]
        mz = 1 * np.outer(np.ones(np.size(u)), np.cos(v)) + mouCC[1]

        ax.plot_surface(mx, my, mz,  rstride=4, cstride=4, color='r', linewidth=0, alpha=0.5)

        yy, zz = np.meshgrid(range(2), range(2))
        xx = yy*0
        
        ax.plot_surface(xx, yy, zz)

        yy1, xx1 = np.meshgrid(range(2), range(2))
        zz1 = yy1*0
        
        ax.plot_surface(xx1, yy1, zz1)

        yy, zz = np.meshgrid(range(25), range(7))
        xx = yy*0
        
        ax.plot_surface(xx, yy, zz)

        yy1, xx1 = np.meshgrid(range(25), range(10))
        zz1 = yy1*0
        
        ax.plot_surface(xx1, yy1, zz1)
        
        ax.set_xlabel('X Label')    
        ax.set_ylabel('Z Label')
        ax.set_zlabel('Y Label')

        ### Set the plot angle and view     ###
        ax.view_init(elev=10., azim=-80)

        plt.xlim((-1.5*refCC[0],refCC[0]*1.5))
        plt.ylim((0,refCC[2]*1.2))
        plt.gca().invert_xaxis()
        
        plt.draw()
        #plt.show(block=False)
        plt.pause(0.0001) 

        print("Drawing new frame")
    else:
        fig = plt.figure(figsize=(12,12), dpi=300)
        ax = fig.add_subplot(111, projection='3d')
        #ax.set_aspect('equal')

        x1 = 1 * np.outer(np.cos(u), np.sin(v))
        y1 = 1 * np.outer(np.sin(u), np.sin(v))
        z1 = 1 * np.outer(np.ones(np.size(u)), np.cos(v))

        ax.plot_surface(x1, y1, z1,  rstride=4, cstride=4, color='k', linewidth=0, alpha=0.5)
        
        x = 1 * np.outer(np.cos(u), np.sin(v)) + refCC[0]
        y = 1 * np.outer(np.sin(u), np.sin(v)) + refCC[2]               # Since we're plotting the y on the z-axis and vice versa
        z = 1 * np.outer(np.ones(np.size(u)), np.cos(v)) + refCC[1]     # Since we're plotting the y on the z-axis and vice versa

        ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='b', linewidth=0, alpha=0.5)

        mx = 1 * np.outer(np.cos(u), np.sin(v)) + mouCC[0]
        my = 1 * np.outer(np.sin(u), np.sin(v)) + mouCC[2]
        mz = 1 * np.outer(np.ones(np.size(u)), np.cos(v)) + mouCC[1]

        ax.plot_surface(mx, my, mz,  rstride=4, cstride=4, color='r', linewidth=0, alpha=0.5)

        yy, zz = np.meshgrid(range(25), range(7))
        xx = yy*0
        
        ax.plot_surface(xx, yy, zz)

        yy1, xx1 = np.meshgrid(range(25), range(10))
        zz1 = yy1*0
        
        ax.plot_surface(xx1, yy1, zz1)
        
        ax.set_xlabel('X Label')    
        ax.set_ylabel('Z Label')
        ax.set_zlabel('Y Label')

        ### Set the plot angle and view     ###
        ax.view_init(elev=10., azim=-80)

        plt.ion()
        plt.xlim((-1.5*refCC[0],refCC[0]*1.5))
        plt.ylim((0,refCC[2]*1.2))
        plt.gca().invert_xaxis()

        print("Starting plot")
        plt.draw()
        #plt.show(block=False)
        plt.pause(0.0001) 
        
    if cv2.waitKey(100) == 13 :
        break

video.release()
cv2.destroyAllWindows() 

