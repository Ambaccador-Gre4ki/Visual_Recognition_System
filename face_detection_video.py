from mtcnn.mtcnn import MTCNN
import cv2
import numpy
import imutils

detector = MTCNN()
capture = cv2.VideoCapture('samples/detection_video/input/miss-russia.mp4')

frame_id = 0
face_n = 0

while True:
    frame_id += 1
    success, frame = capture.read()

    if success:
        if frame.shape[0] < frame.shape[1]:
            frame = imutils.resize(frame, height=1000)
        else:
            frame = imutils.resize(frame, width=1000)

        image_size = numpy.asarray(frame.shape)[0:2]
        faces_boxes = detector.detect_faces(frame)
        image_detected = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        marked_color = (0, 255, 0, 1)

        if faces_boxes:
             for face_box in faces_boxes:
                # Увеличение счётчика файлов
                face_n += 1
                # Координаты лица
                x, y, w, h = face_box['box']

                # Отступы для увеличения рамки
                d = h - w  # Разница между высотой и шириной
                w = w + d  # Делаем изображение квадратным
                x = numpy.maximum(x - round(d/2), 0)
                x1 = numpy.maximum(x - round(w/4), 0)
                y1 = numpy.maximum(y - round(h/4), 0)
                x2 = numpy.minimum(x + w + round(w/4), image_size[1])
                y2 = numpy.minimum(y + h + round(h/4), image_size[0])

                # Получение картинки с лицом
                cropped = image_detected[y1:y2, x1:x2, :]
                face_image = cv2.resize(cropped, (160, 160), interpolation=cv2.INTER_AREA)

                # Имя файла (уверенность + номер)
                face_filename = str(face_box['confidence']) + '.' + str(face_n) + '.jpg'

                # Отборка лиц {selected|rejected}
                if face_box['confidence'] > 0.99:  # 0.99 - уверенность сети в процентах что это лицо

                    # Рисует белый квадрат на картинке по координатам
                    cv2.rectangle(
                        image_detected,
                        (x1, y1),
                        (x2, y2),
                        (255, 255, 255, 1),
                        1
                    )

                    # Путь к директории с качественными изображениями
                    face_path = 'samples/detection_video/output/faces/selected/' + face_filename

                    # Информируем консоль
                    print('\033[92m' + face_filename + '\033[0m')

                else:

                    # Рисует красный квадрат на картинке по координатам
                    cv2.rectangle(
                        image_detected,
                        (x1, y1),
                        (x2, y2),
                        (0, 0, 255, 1),
                        1
                    )

                    # Путь к директории с отбракованными изображениями
                    face_path = 'samples/detection_video/output/faces/rejected/' + face_filename

                    # Информируем консоль
                    print('\033[91m' + face_filename + '\033[0m')

                # Сохранение изображения лица на диск в директории {selected|rejected}
                cv2.imwrite(face_path, face_image)

        # Сохраняем кадр с видео
        cv2.imwrite('samples/detection_video/output/frames/' + str(frame_id) + '.jpg', image_detected)

    else:
        break           