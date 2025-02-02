import cv2
import numpy as np

# Função para calcular a variação de textura e reflexos
def analyze_texture_and_reflection(roi):
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Variância do Laplaciano (textura)
    laplacian_var = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
    
    # Brilho médio (reflexos)
    mean_brightness = np.mean(gray_roi)
    
    return laplacian_var, mean_brightness

# Função para simular análise de profundidade com gradientes
def simulate_depth_analysis(roi):
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Gradiente horizontal e vertical
    grad_x = cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_roi, cv2.CV_64F, 0, 1, ksize=3)
    
    # Magnitude do gradiente
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    depth_score = np.mean(gradient_magnitude)  # Média dos gradientes
    
    return depth_score

# Inicializar captura da câmera
cap = cv2.VideoCapture(2)

# Configurar detecção facial
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Criar janela única
cv2.namedWindow("Detecção de Vivacidade", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar o quadro da câmera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extrair ROI do rosto detectado
        roi_frame = frame[y:y+h, x:x+w]

        # Analisar textura e reflexos
        laplacian_var, mean_brightness = analyze_texture_and_reflection(roi_frame)

        # Simular análise de profundidade
        depth_score = simulate_depth_analysis(roi_frame)

        # Decidir se é uma pessoa viva ou uma imagem
        if laplacian_var < 50 or mean_brightness > 250 or depth_score < 10:
            label = "Dead"
            color = (0, 0, 255)  # Vermelho
        else:
            label = "Life"
            color = (0, 255, 0)  # Verde

        # Desenhar rótulo e caixa ao redor do rosto detectado
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2)

    # Mostrar o vídeo com as detecções
    cv2.imshow("Detecção de Vivacidade", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
