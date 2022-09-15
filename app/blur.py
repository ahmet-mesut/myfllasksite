

import torch
import cv2

trained_model = '/app/app/best.pt'
#C:/Users/Murat/Desktop/non -labeled/herokuflask/myfllasksite/app/best.pt
#my_model2 = Licence_Plate(trained_model)


class Licence_Platee:

      def __init__(self,trained_model):

            self.model = torch.hub.load('ultralytics/yolov5', 'custom', trained_model)

      def detect(self,img):

            self.result_model = self.model(img)
            self.img = img
            img = self.plot()

            return img

      def plot(self):

            det = self.result_model.xyxy[0]
            if len(det) == 0:

                  font                   = cv2.FONT_HERSHEY_TRIPLEX
                  bottomLeftCornerOfText = (0,85)
                  fontScale              = 3
                  fontColor              = (0,0,255)
                  thickness              = 3

                  img = cv2.putText(self.img,'Hello World!', bottomLeftCornerOfText,font,fontScale,fontColor,thickness)

                  return img
            else:
                  for i in range(len(det)):
                       x = int(det[i][0])
                       y = int(det[i][1])
                       w = int(det[i][2])-x
                       h = int(det[i][3])-y
                       img = self.img
                       img[y:y+h, x:x+w]=cv2.blur(img[y:y+h, x:x+w],(23,23), 0)
                      #pt1 = (int(det[i][0]), int(det[i][1]))
                      #pt2 = (int(det[i][2]), int(det[i][3]))
                      #img = cv2.rectangle(self.img,pt1,pt2,color=(0,255,0),thickness=2)

                  return img

my_model_blur = Licence_Platee(trained_model)
