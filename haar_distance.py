import cv2
Known_distance = 13  # Inches
Known_width = 1.2  # Inches

GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLACK = (0, 0, 0)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
CYAN = (255, 255, 0)
MAGENTA = (255, 0, 242)
GOLDEN = (32, 218, 165)
LIGHT_BLUE = (255, 9, 2)
PURPLE = (128, 0, 128)
CHOCOLATE = (30, 105, 210)
PINK = (147, 20, 255)
ORANGE = (0, 69, 255)

fonts = cv2.FONT_HERSHEY_COMPLEX
fonts2 = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
fonts3 = cv2.FONT_HERSHEY_COMPLEX_SMALL
fonts4 = cv2.FONT_HERSHEY_TRIPLEX

cap = cv2.VideoCapture(0)
Distance_level = 0

face_detector = cv2.CascadeClassifier("stopsign_classifier.xml")

def FocalLength(measured_distance, real_width, width_in_rf_image):
    focal_length = (width_in_rf_image * measured_distance) / real_width
    return focal_length

def Distance_finder(Focal_Length, real_face_width, face_width_in_frame):
    distance = (real_face_width * Focal_Length)/face_width_in_frame
    return distance

def face_data(image, CallOut, Distance_level):

    face_width = 0
    face_x, face_y = 0, 0
    face_center_x = 0
    face_center_y = 0
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_image, 1.3, 5)
    for (x, y, w, h) in faces:
        line_thickness = 2
        LLV = int(h*0.12)

        cv2.rectangle(image, (x, y), (x+w, y+h), RED, 2)

        face_width = w
        face_center = []
        face_center_x = int(w/2)+x
        face_center_y = int(h/2)+y
        if Distance_level < 10:
            Distance_level = 10
            
    return face_width, faces, face_center_x, face_center_y


ref_image = cv2.imread("ref_image.png")

ref_image_face_width, _, _, _ = face_data(ref_image, False, Distance_level)
Focal_length_found = FocalLength(Known_distance, Known_width, ref_image_face_width)

while True:
    _, frame = cap.read()

    face_width_in_frame, Faces, FC_X, FC_Y = face_data(frame, True, Distance_level)
    
    for (face_x, face_y, face_w, face_h) in Faces:
        if face_width_in_frame != 0:

            Distance = Distance_finder(Focal_length_found, Known_width, face_width_in_frame)
            Distance = round(Distance, 2)
            
            Distance_level = int(Distance)

            cv2.putText(frame, f"Distance {Distance} Inches",(face_x-6, face_y-6), fonts, 1, (BLACK), 2)
            
    cv2.imshow("frame", frame)
    

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
