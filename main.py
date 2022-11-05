# import the object detection class
from imageai.Detection import ObjectDetection

import cv2

# create an instance/object of the object detection class
obj_detect = ObjectDetection()

# set the model type for object detection
obj_detect.setModelTypeAsYOLOv3()

# load the model
obj_detect.setModelPath(r"./yolo.h5")
obj_detect.loadModel()

# set up the video capture, 0 at the params means the default cam that's being used in your computer
cam = cv2.VideoCapture(0)

# define the width and length for the frame of the cam capture
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 650)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 750)

while True:
    ret, img = cam.read()
    annotated_image, preds = obj_detect.detectObjectsFromImage(input_image=img,
                    input_type="array",
                      output_type="array",
                      display_percentage_probability=False,
                      display_object_name=True)

    cv2.imshow("", annotated_image)

    if (cv2.waitKey(1) & 0xFF == ord("q")) or (cv2.waitKey(1)==27):
        break

cam.release()
cv2.destroyAllWindows()