import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import find_peaks

# Inicializar o MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Configurar a detecção facial
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Função para calcular batimentos cardíacos a partir do canal vermelho (R)
def calculate_heartbeat(red_signal, fps):
    if len(red_signal) < fps * 5:  # Pelo menos 5 segundos de dados necessários
        return None
    
    # Normalizar o sinal
    normalized_signal = (red_signal - np.mean(red_signal)) / np.std(red_signal)
    
    # Detectar picos no sinal
    peaks, _ = find_peaks(normalized_signal, distance=fps // 2)  # Frequência mínima de 30 BPM
    
    # Calcular BPM
    time_interval = len(normalized_signal) / fps
    bpm = len(peaks) / time_interval * 60
    return bpm

# Função para analisar textura e reflexos
def analyze_texture_and_reflection(roi):
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Variância do Laplaciano (textura)
    laplacian_var = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
    
    # Brilho médio (reflexos)
    mean_brightness = np.mean(gray_roi)
    
    return laplacian_var, mean_brightness

# Inicializar captura da câmera
cap = cv2.VideoCapture(2)

cv2.namedWindow("Detecção Facial com MediaPipe", cv2.WINDOW_NORMAL)

fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps == 0:
    fps = 30  # Valor padrão caso o FPS não seja detectado

red_signal = []  # Buffer para armazenar o canal vermelho (R)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar o quadro da câmera.")
        break

    # Converter o quadro para RGB (necessário para MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Processar o quadro para detecção facial
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            # Obter coordenadas do rosto detectado
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            
            # Recortar a região do rosto (ROI)
            roi_frame = frame[y:y+h, x:x+w]
            
            if roi_frame.size > 0:  # Certificar-se de que a ROI não está vazia
                # Analisar textura e reflexos na ROI
                laplacian_var, mean_brightness = analyze_texture_and_reflection(roi_frame)

                # Adicionar o valor médio do canal vermelho ao buffer de sinal
                red_channel_mean = np.mean(roi_frame[:, :, 2])  # Canal vermelho (R)
                red_signal.append(red_channel_mean)

                # Manter apenas os últimos 10 segundos de dados no buffer
                if len(red_signal) > fps * 10:
                    red_signal.pop(0)

                # Calcular batimentos cardíacos se houver dados suficientes
                bpm = calculate_heartbeat(np.array(red_signal), fps)

                # Decidir se é um rosto "vivo" ou "não vivo"
                if laplacian_var < 50 or mean_brightness > 200:
                    label = "Não Vivo"
                    color = (0, 0, 255)  # Vermelho
                else:
                    label = "Vivo"
                    color = (0, 255, 0)  # Verde

                if bpm is not None and label == "Vivo":
                    label += f" | BPM: {bpm:.2f}"

                # Desenhar retângulo e exibir rótulo no rosto detectado
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, color, 2)

            # # Desenhar a detecção usando MediaPipe Drawing Utils
            mp_drawing.draw_detection(frame, detection)

    # Mostrar o vídeo com as detecções
    cv2.imshow("Detecção Facial com MediaPipe", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
