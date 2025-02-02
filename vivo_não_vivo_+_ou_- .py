import cv2
import numpy as np
from scipy.signal import find_peaks

# Função para extrair sinais RGB médios da região do rosto
def extract_rgb_signals(frame, roi):
    x, y, w, h = roi  # Região de interesse (ROI)
    roi_frame = frame[y:y+h, x:x+w]
    b, g, r = cv2.split(roi_frame)  # Separar os canais RGB
    return np.mean(r), np.mean(g), np.mean(b)

# Função para processar o sinal e detectar batimentos cardíacos
def detect_heartbeat(signal, fps):
    signal = (signal - np.mean(signal)) / np.std(signal)  # Normalizar o sinal
    peaks, _ = find_peaks(signal, distance=fps//2)  # Detectar picos no sinal
    time_interval = len(signal) / fps
    bpm = len(peaks) / time_interval * 60  # Calcular BPM
    return bpm

# Função para analisar textura e reflexos na ROI
def detect_texture_and_reflection(roi):
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray_roi, cv2.CV_64F).var()  # Variância do Laplaciano (textura)
    mean_brightness = np.mean(gray_roi)  # Brilho médio (reflexos)
    return laplacian_var, mean_brightness

# Função para analisar variação dinâmica ao longo do tempo
def analyze_variation_in_color_and_brightness(roi_sequence):
    brightness_values = []
    for roi in roi_sequence:
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray_roi)
        brightness_values.append(mean_brightness)
    brightness_variation = np.std(brightness_values)  # Variação no brilho
    return brightness_variation

# Inicializar captura da câmera
cap = cv2.VideoCapture(2)

# Configurações iniciais
fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps == 0:
    fps = 30  # Valor padrão caso o FPS não seja detectado

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
red_signal = []
roi_sequence = []  # Sequência de ROIs para análise dinâmica
label = "Desconhecido"

# Criar uma única janela para exibição
cv2.namedWindow("Detecção de Vivacidade", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar o quadro da câmera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Converter para escala de cinza
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Desenhar a caixa ao redor do rosto detectado
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Extrair ROI atual para análise
        roi_frame = frame[y:y+h, x:x+w]

        # Analisar textura e reflexos
        laplacian_var, mean_brightness = detect_texture_and_reflection(roi_frame)

        # Adicionar ROI à sequência para análise dinâmica
        roi_sequence.append(roi_frame)
        if len(roi_sequence) > fps * 3:  # Máximo de 3 segundos de dados
            roi_sequence.pop(0)

        if len(roi_sequence) >= fps:  # Pelo menos 1 segundo de dados para análise dinâmica
            brightness_variation = analyze_variation_in_color_and_brightness(roi_sequence)

            # Decidir se é uma foto ou um rosto real com base nas análises combinadas
            if laplacian_var < 50 or mean_brightness > 200 or brightness_variation < 5:
                label = "Não Vivo"
            else:
                label = "Vivo"

                # Extrair sinais RGB apenas da região do rosto para batimentos cardíacos
                r_mean, g_mean, b_mean = extract_rgb_signals(frame, (x, y, w, h))
                red_signal.append(r_mean)

                if len(red_signal) > fps * 10:  # Máximo de 10 segundos de dados
                    red_signal.pop(0)

                if len(red_signal) >= fps * 5:  # Pelo menos 5 segundos de dados
                    bpm = detect_heartbeat(np.array(red_signal), fps)
                    label += f" | BPM: {bpm:.2f}"

        # Exibir rótulo na face detectada ("Vivo" ou "Não Vivo")
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0) if "Vivo" in label else (0, 0, 255), 2)

    # Atualizar a mesma janela com o novo frame processado
    cv2.imshow("Detecção de Vivacidade", frame)

    # Verificar se a tecla 'q' foi pressionada para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
