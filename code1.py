import cv2
import easyocr
from imutils.video import VideoStream

# Инициализируем камеру
vs = VideoStream(src=0).start()

# Дождитесь, пока камера загрузится
# Может потребоваться время, чтобы камера инициализировалась
# или вы можете указать конкретное устройство src, если у вас их несколько

# Инициализируем EasyOCR
reader = easyocr.Reader(['en'])

while True:
    # Получаем кадр с камеры
    frame = vs.read()

    # Преобразуем кадр в оттенки серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Применяем фильтры для улучшения читаемости текста (настройте по необходимости)
    # Пример:
    # gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, (5, 5))

    # Распознаем текст на кадре с помощью EasyOCR
    results = reader.readtext(gray)

    for (bbox, text, prob) in results:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))

        # Фильтруем короткие строки (меньше определенной длины, например, 5 символов)
        if len(text) > 5:
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(frame, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Отображаем видеопоток
    cv2.imshow("License Plate Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Нажмите "Esc" для выхода
        break

# Очистка
cv2.destroyAllWindows()
vs.stop()
