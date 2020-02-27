from __future__ import division
import dlib
import cv2
import numpy as np
import math

eyeRealDist = 2.656 #distance between eyes
focal = 734.94 #focal point of C270 camera

PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat' 
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

class facialFeat():
    
    def __init__(self, camNum):
        self.camNum = camNum
    
    def toArray(self, shape, dtype='int'):
        coords = np.zeros((68,2), dtype=dtype) # initialize blank np array for 68 features
        
        for i in range(0,68):
            coords[i] = (shape.part(i).x, shape.part(i).y) # insert the x and y coords of the shape into the np array
        
        return coords
        
    def trackFaceCont(self):
        cap = cv2.VideoCapture(self.camNum)
        
        while True:
            ret, img = cap.read() 
            if ret == False:
                print("Failed to read image from camera \n") 
                break
            
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_small = cv2.resize(img_gray, (0,0), fx=0.75, fy=0.75) # downsize image for more effiecient processing
            
            face = detector(img_small, 0) # use default detector to find faces in frame
            
            if len(face) == 1: # only if one face is detected
                shape = predictor(img_small, face[0])
                shape_np = self.toArray(shape)
                
                # index 36 - right corner of right eye
                # index 42 - right corner of left eye
                eyedistance = math.sqrt(math.pow((4/3)*shape_np[36][0] - (4/3)*shape_np[42][0],2) + math.pow((4/3)*shape_np[36][1] - (4/3)*shape_np[42][1],2))
                mouthDist = round(((focal * eyeRealDist) / eyedistance),3)
                # using perspective geometry to find mouth distance from eye distance
                
                mouthPosX = int((shape_np[51][0] + shape_np[57][0])*(2/3)) - 320
                mouthPosY = int((shape_np[51][1] + shape_np[57][1])*(2/3)) - 240
                # indeces 51 and 57 are the top and bottom of the middle of the mouth, respectively
                
                
                lipDist1 = math.sqrt(math.pow((4/3)*shape_np[61][0] - (4/3)*shape_np[67][0],2) + math.pow((4/3)*shape_np[61][1] - (4/3)*shape_np[67][1],2))
                lipDist2 = math.sqrt(math.pow((4/3)*shape_np[62][0] - (4/3)*shape_np[66][0],2) + math.pow((4/3)*shape_np[62][1] - (4/3)*shape_np[66][1],2))
                lipDist3 = math.sqrt(math.pow((4/3)*shape_np[63][0] - (4/3)*shape_np[65][0],2) + math.pow((4/3)*shape_np[63][1] - (4/3)*shape_np[65][1],2))
                lipDistBool = ((lipDist1 + lipDist2 + lipDist3)/3)/eyedistance > 0.15
                # using the six different landmarks for the inner middle of the mouth to find the distance between lips
                # 0.10 - 0.02: inverse range of distance within reasonable operating range of robot
                
                for i, (x,y) in enumerate(shape_np):
                    cv2.putText(img, str(i), (int(x*(4/3)), int(y*(4/3))), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0,255,0), 1,cv2.LINE_AA)
                    cv2.circle(img, (int(x*(4/3)), int(y*(4/3))), 1, (255,255,255), -1)
                # labeling the landmarks on the screen
                    
                cv2.putText(img, "Mouth open: " + str(lipDistBool), (25, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),1,cv2.LINE_AA)
                cv2.putText(img, "Distance from camera: " + str(mouthDist) + " inches", (25, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),1,cv2.LINE_AA)
                cv2.putText(img, "Mouth position: (" + str(mouthPosX) + ", " + str(mouthPosY) + ")", (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
                # outputting various analytics to the screen
                        
            cv2.imshow('img',img)
            
            k = cv2.waitKey(30) & 0xFF
            if k == 27:
                cap.release()
                cv2.destroyAllWindows()
                break
            
    def trackFaceFrame(self): #returns x, y, dist, isOpen
        cap = cv2.VideoCapture(self.camNum)
        
        ret, img = cap.read() 
        if ret == False:
            print("Failed to read image from camera \n") 
            return -1, -1, -1, False
        
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_small = cv2.resize(img_gray, (0,0), fx=0.75, fy=0.75) # downsize image for more effiecient processing
        
        face = detector(img_small, 0) # use default dlib detector to find faces in frame
        
        cap.release()

        if len(face) == 1: # only if one face is detected
            shape = predictor(img_small, face[0])
            shape_np = self.toArray(shape)
            
            # index 36 - right corner of right eye
            # index 42 - right corner of left eye
            eyedistance = math.sqrt(math.pow((4/3)*shape_np[36][0] - (4/3)*shape_np[42][0],2) + math.pow((4/3)*shape_np[36][1] - (4/3)*shape_np[42][1],2))
            mouthDist = round(((focal * eyeRealDist) / eyedistance),3)
            # using perspective geometry to find mouth distance from eye distance
            
            mouthPosX = int((shape_np[51][0] + shape_np[57][0])*(2/3))
            mouthPosY = int((shape_np[51][1] + shape_np[57][1])*(2/3))
            # indeces 51 and 57 are the top and bottom of the middle of the mouth, respectively
            
            lipDist1 = math.sqrt(math.pow((4/3)*shape_np[61][0] - (4/3)*shape_np[67][0],2) + math.pow((4/3)*shape_np[61][1] - (4/3)*shape_np[67][1],2))
            lipDist2 = math.sqrt(math.pow((4/3)*shape_np[62][0] - (4/3)*shape_np[66][0],2) + math.pow((4/3)*shape_np[62][1] - (4/3)*shape_np[66][1],2))
            lipDist3 = math.sqrt(math.pow((4/3)*shape_np[63][0] - (4/3)*shape_np[65][0],2) + math.pow((4/3)*shape_np[63][1] - (4/3)*shape_np[65][1],2))
            lipDistBool = ((lipDist1 + lipDist2 + lipDist3)/3)/eyedistance > 0.15
            # using the six different landmarks for the inner middle of the mouth to find the distance between lips
            # 0.10 - 0.02: inverse range of distance within reasonable operating range of robot
            
            return mouthPosX, mouthPosY, mouthDist, lipDistBool
        
        else:
            return -2, -2, -2, False
        
            
#faceTrack = facialFeat(1)
#faceTrack.trackFaceCont()


# 0-5 right jaw (top to bottom)
# 6-10 chin (right to left)
# 11-16 left jaw (bottom to top)
# 17-21 right eyebrow (right to left)
# 22-26 left eyebrock (right to left)
# 27-30 nose bridge (top to bottom)
# 31-35 under nose (right to left)
# 36-39 top of right eye including corners (right to left)
# 40-41 bottom of right yee excluding corners (left to right)
# 42-45 top of left eye including corners (right to left)
# 46-47 bottom of left eye excluding corners (left to right)
# 48-54 top of outside mouth including corners (right to left)
# 55-59 bottom of outside mouth excluding corners (left to right)
# 60 right inner mouth corner
# 61-63 inner mouth top (right to left)
# 64 left inner mouth corner
# 65-67 inner mouth bottom (left to right)