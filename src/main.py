import cv2
import dlib
import numpy as np
import pygame
import threading

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = np.linalg.norm(mouth[3] - mouth[9])
    B = np.linalg.norm(mouth[2] - mouth[10])
    C = np.linalg.norm(mouth[4] - mouth[8])
    D = np.linalg.norm(mouth[0] - mouth[6])
    return (A + B + C) / (3.0 * D)

# Caminho para o arquivo shape_predictor_68_face_landmarks.dat
predictor_path = 'models/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

left_eye_indices = list(range(36, 42))
right_eye_indices = list(range(42, 48))
mouth_indices = list(range(48, 68))

cap = cv2.VideoCapture(1)

# Configurações para a detecção de fadiga e bocejo
ear_threshold = 0.25
mar_threshold = 0.60
consecutive_frames_threshold = 20
frame_count = 0

# Inicializar o pygame mixer
pygame.mixer.init()
alarm_sound = 'audio/alarm.wav'
alarm = pygame.mixer.Sound(alarm_sound)
alarm_playing = False

def play_alarm():
    global alarm_playing
    alarm_playing = True
    while alarm_playing:
        alarm.play(maxtime=3000)  # Toca o alarme por 3 segundos

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    fatigue_detected = False
    yawn_detected = False

    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = np.array([(p.x, p.y) for p in landmarks.parts()])

        # Desenhando o contorno do rosto
        for i in range(0, 68):
            cv2.circle(frame, (landmarks[i][0], landmarks[i][1]), 2, (0, 255, 0), -1)

        # Desenhando os olhos
        left_eye = landmarks[left_eye_indices]
        right_eye = landmarks[right_eye_indices]
        cv2.polylines(frame, [np.int32(left_eye)], isClosed=True, color=(0, 255, 255), thickness=1)
        cv2.polylines(frame, [np.int32(right_eye)], isClosed=True, color=(0, 255, 255), thickness=1)

        # Desenhando a boca
        mouth = landmarks[mouth_indices]
        cv2.polylines(frame, [np.int32(mouth)], isClosed=True, color=(255, 0, 0), thickness=1)

        # Calculando a razão entre a altura e a largura dos olhos
        left_eye_ratio = eye_aspect_ratio(left_eye)
        right_eye_ratio = eye_aspect_ratio(right_eye)
        ear = (left_eye_ratio + right_eye_ratio) / 2.0

        # Calculando a razão da boca
        mar = mouth_aspect_ratio(mouth)

        if ear < ear_threshold or mar > mar_threshold:
            frame_count += 1
            if frame_count >= consecutive_frames_threshold:
                fatigue_detected = True
                yawn_detected = True
        else:
            frame_count = 0

    if fatigue_detected or yawn_detected:
        cv2.putText(frame, "Fadiga detectada!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        if not alarm_playing:
            alarm_thread = threading.Thread(target=play_alarm)
            alarm_thread.start()
    else:
        if alarm_playing:
            alarm_playing = False
            cv2.putText(frame, "Alerta desligado", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Detector de Fadiga", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
