from mtcnn.mtcnn import MTCNN
import cv2
import numpy
import imutils

detector = MTCNN() #создаем сеть нахождения лиц
image = cv2.imread('samples/detection_image/input/people.jpg') #загрузка картинки с лицами

if image.shape[0] < image.shape[1]:
    image = imutils.resize(image, height = 1000)
else:
    image = imutils.resize(image, width = 1000)

# получаем размеры изображения
image_size = numpy.asarray(image.shape)[0:2]

# получение списка лиц с координатами и значением уверенности
faces_boxes = detector.detect_faces(image)

# копия изображения, чтобы нарисовать рамки
image_detected = image.copy()

# копия изображения для рисования меток
image_marked = image.copy()

# замена цветокоррекции на rgb
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

marked_color = (0, 255, 0, 1) # цвет меток

if faces_boxes:
    face_n = 0 # счётчик
    for face_box in faces_boxes:
        face_n += 1
        x, y, w, h = face_box['box'] # координаты лица

        d = h - w # разница между высотой и шириной
        w += d # КВАДРАТизация рамки
        x = numpy.maximum(x - round(d/2), 0)
        x1 = numpy.maximum(x - round(w/4), 0)
        y1 = numpy.maximum(x - round(h/4), 0)
        x2 = numpy.minimum(x + w + round(w/4), image_size[1])
        y2 = numpy.minimum(y + h + round(h/4), image_size[0])

        print(x1, y1, x2, y2, image_size[0], image_size[1])

        cropped = image_detected[y1:y2, x1:x2, :]
        face_image = cv2.resize(cropped, (160, 160), interpolation=cv2.INTER_AREA)

        face_filename = str(face_box['confidence']) + '.' + str(face_n) + '.jpg'

        if face_box['confidence'] > 0.99:
            cv2.rectangle(image_detected, (x1, y1), (x2, y2), (255, 255, 255, 1), 1)

            face_path = 'samples/detection_image/output/faces/selected/' + face_filename
            print('\033[92m' + face_filename + '\033[0m')

        else:
            cv2.rectangle(image_detected, (x1, y1), (x2, y2),
                          (0, 0, 255, 255, 1), 1)
            face_path = 'samples/detection_image/output/faces/rejected/' + face_filename
            print('\033[91m' + face_filename + '\033[0m')

        cv2.rectangle(image_marked,
                      (face_box['keypoints']['left_eye'][0], face_box['keypoints']['left_eye'][1]),
                      (face_box['keypoints']['left_eye'][0] + 1, face_box['keypoints']['left_eye'][1] + 1),
                      marked_color, 1)
        
        cv2.rectangle(image_marked,
                      (face_box['keypoints']['right_eye'][0], face_box['keypoints']['right_eye'][1]),
                      (face_box['keypoints']['right_eye'][0] + 1, face_box['keypoints']['right_eye'][1] + 1),
                      marked_color, 1)
        
        cv2.rectangle(image_marked,
                      (face_box['keypoints']['nose'][0], face_box['keypoints']['nose'][1]),
                      (face_box['keypoints']['nose'][0] + 1, face_box['keypoints']['nose'][1] + 1),
                      marked_color, 1)
        
        cv2.rectangle(image_marked,
                      (face_box['keypoints']['mouth_left'][0], face_box['keypoints']['mouth_left'][1]),
                      (face_box['keypoints']['mouth_left'][0] + 1, face_box['keypoints']['mouth_left'][1] + 1),
                      marked_color, 1)
        
        cv2.rectangle(image_marked,
                      (face_box['keypoints']['mouth_right'][0], face_box['keypoints']['mouth_right'][1]),
                      (face_box['keypoints']['mouth_right'][0] + 1, face_box['keypoints']['mouth_right'][1] + 1),
                      marked_color, 1)
        
        cv2.imwrite(face_path, face_image)
    
    cv2.imwrite('demo/detection_image/output/people-detected.jpg', image_detected)
    cv2.imwrite('demo/detection_image/output/people-marked.jpg', image_marked)