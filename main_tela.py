import dlib
import cv2
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
import os
import mss
import time
from deepface import DeepFace
from insightface.app import FaceAnalysis

# Configurações globais
FPS_MIN = 30
BUFFER_SECONDS = 2
MOTION_THRESHOLD = 5000
TEXTURE_THRESHOLD = 15
BRIGHTNESS_THRESHOLD = 500

# Suprimir avisos
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
import logging
logging.getLogger('mediapipe').setLevel(logging.ERROR)

def select_region():
    try:
        with mss.mss() as sct:
            monitor = sct.monitors[0]
            
            # Capturar 40% da largura da tela
            width = int(monitor["width"] * 0.36)  # 40% da largura
            height = int(monitor["height"] * 0.9)  # 90% da altura para evitar a faixa preta inferior
            
            # Deslocar a área de captura para a direita em 7% da largura da tela
            left_offset = int(monitor["width"] * 0.07)  # Deslocamento de 7% para a direita

            # Deslocar a área de captura para a esquerda em 30% da largura da tela
            #width_offset = int(monitor["width"] * 0.02)  # Deslocamento de 30% para a esquerda
            
            return {
                "left": monitor["left"] + left_offset,
                "top": monitor["top"],
                "width": width, # + width_offset,
                "height": height
            }
    except Exception as e:
        print(f"Erro ao obter dimensões do monitor: {e}")
        return None

def capture_region(sct, region):
    try:
        # Configurar região para captura
        monitor = {
            "top": region["top"],
            "left": region["left"],
            "width": region["width"],
            "height": region["height"]
        }
        
        # Capturar região específica
        screenshot = sct.grab(monitor)
        
        # Converter para formato OpenCV
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame
    except Exception as e:
        print(f"Erro na captura da região: {e}")
        return None

def train_face_recognizer(known_faces_dir):
    embeddings = {}
    label_map = {}
    app = FaceAnalysis(
        allowed_modules=['detection', 'recognition'],
        providers=['CPUExecutionProvider']
    )
    app.prepare(ctx_id=0, det_size=(640, 480))
    
    for person_name in os.listdir(known_faces_dir):
        person_dir = os.path.join(known_faces_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        
        for filename in os.listdir(person_dir):
            if filename.endswith(('.jpg', '.png', '.jpeg')):
                image_path = os.path.join(person_dir, filename)
                try:
                    img = cv2.imread(image_path)
                    faces = app.get(img)
                    if len(faces) > 0:
                        embedding = faces[0].embedding
                        embeddings[person_name] = embedding
                        label_map[person_name] = person_name
                        print(f"Face processada com sucesso: {filename}")
                    else:
                        print(f"Nenhuma face detectada em: {filename}")
                except Exception as e:
                    print(f"Erro ao processar {filename}: {e}")
                    continue
    
    return embeddings, label_map, app

def calculate_heartbeat(red_signal, fps):
    if len(red_signal) < fps * BUFFER_SECONDS:
        print(f"Buffer insuficiente: {len(red_signal)} frames < {fps * BUFFER_SECONDS} necessários")
        return None
        
    signal = red_signal - np.mean(red_signal)
    signal = signal / np.std(signal)
    
    normalized_signal = signal
    
    peaks, _ = find_peaks(normalized_signal, 
                         distance=int(fps * 0.2),
                         height=0.05,
                         prominence=0.1)
    
    if len(peaks) < 2:
        print(f"Picos insuficientes: {len(peaks)} encontrados")
        return None
        
    time_interval = len(normalized_signal) / fps
    bpm = len(peaks) / time_interval * 60
    
    if 30 <= bpm <= 220:
        print(f"BPM calculado: {bpm:.0f}")
        return bpm
    print(f"BPM fora da faixa: {bpm:.0f}")
    return None

def analyze_texture_and_reflection(roi):
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
    mean_brightness = np.mean(gray_roi)
    return laplacian_var, mean_brightness

def detect_motion(previous_roi, current_roi):
    if previous_roi is None:
        return True

    if previous_roi.shape != current_roi.shape:
        current_roi = cv2.resize(current_roi, (previous_roi.shape[1], previous_roi.shape[0]))

    diff = cv2.absdiff(previous_roi, current_roi)
    motion_score = np.sum(diff)
    return motion_score > MOTION_THRESHOLD

def verify_face(frame, reference_img):
    try:
        result = DeepFace.verify(
            frame, 
            reference_img,
            model_name="VGG-Face",  # Outras opções: Facenet, OpenFace, DeepID
            detector_backend="retinaface"  # Outras opções: mtcnn, opencv, ssd
        )
        return result['verified'], result['distance']
    except Exception as e:
        print(f"Erro na verificação: {e}")
        return False, None

def recognize_face(frame, app, embeddings, threshold=0.6):
    try:
        # Detectar faces no frame atual
        faces = app.get(frame)
        if len(faces) == 0:
            return None, None
        
        # Pegar a primeira face detectada
        face = faces[0]
        face_embedding = face.embedding
        
        # Comparar com faces conhecidas
        best_match = None
        best_score = -1
        second_best_score = -1
        
        # Primeiro passo: encontrar os dois melhores matches
        for name, ref_embedding in embeddings.items():
            similarity = np.dot(face_embedding, ref_embedding)
            if similarity > best_score:
                second_best_score = best_score
                best_score = similarity
                best_match = name
            elif similarity > second_best_score:
                second_best_score = similarity
        
        # Segundo passo: verificar se o melhor match é significativamente melhor que o segundo
        if best_score > threshold:
            score_diff = best_score - second_best_score
            if score_diff > 0.50:  # Aumentar a diferença mínima necessária
                return best_match, best_score
        
        return None, None
    except Exception as e:
        print(f"Erro no reconhecimento: {e}")
        return None, None

def main():
    # Usar dlib para detecção de faces com HOG
    hog_face_detector = dlib.get_frontal_face_detector()

    # Inicializar reconhecedor facial
    known_faces_dir = "known_faces"
    embeddings, label_map, face_app = train_face_recognizer(known_faces_dir)
    
    if embeddings is None:
        print("Erro no reconhecedor facial. Executando apenas detecção...")
        use_recognition = False
    else:
        use_recognition = True

    red_signal_per_roi = {}
    previous_rois = {}
    last_face_position = None

    # Selecionar região da tela
    region = select_region()
    if region is None:
        print("Erro ao selecionar região. Encerrando...")
        return

    fps = 30
    frame_delay = 1.0 / fps
    last_frame_time = 0

    # Criar janela simples
    cv2.namedWindow("Face Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Detecção Facial", 800, 600)

    # Inicializar mss fora do loop
    sct = mss.mss()

    try:
        while True:
            current_time = time.time()
            elapsed = current_time - last_frame_time
            
            if elapsed < frame_delay:
                time.sleep(0.001)
                continue
                
            last_frame_time = current_time
            
            try:
                frame = capture_region(sct, region)
                if frame is None:
                    continue
                
                scale_percent = 50
                width = int(frame.shape[1] * scale_percent / 100)
                height = int(frame.shape[0] * scale_percent / 100)
                frame = cv2.resize(frame, (width, height))

                frame_height, frame_width = frame.shape[:2]
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Usar HOG para detecção de faces
                faces_hog = hog_face_detector(gray, 1)
                
                faces = [(d.left(), d.top(), d.width(), d.height()) for d in faces_hog]

                if len(faces) > 0:
                    (x, y, w, h) = faces[0]
                    
                    face_ratio = w / frame_width
                    if face_ratio < 0.2:
                        status = "Aproxime-se"
                    elif face_ratio > 0.4:
                        status = "Afaste-se"
                    else:
                        status = "Distancia OK"

                    buffer_key = 'main_face'
                    
                    roi_color = frame[y:y+h, x:x+w]
                    roi_gray = gray[y:y+h, x:x+w]

                    motion_detected = detect_motion(previous_rois.get(buffer_key), roi_gray)
                    previous_rois[buffer_key] = roi_gray.copy()

                    laplacian_var, mean_brightness = analyze_texture_and_reflection(roi_color)

                    roi_center = roi_color
                    mean_colors = np.mean(roi_center, axis=(0,1))
                    signal_value = np.mean(mean_colors)
                    
                    if buffer_key not in red_signal_per_roi:
                        red_signal_per_roi[buffer_key] = []
                    
                    red_signal_per_roi[buffer_key].append(signal_value)

                    max_buffer = int(fps * BUFFER_SECONDS * 1.1)
                    if len(red_signal_per_roi[buffer_key]) > max_buffer:
                        red_signal_per_roi[buffer_key] = red_signal_per_roi[buffer_key][-max_buffer:]

                    bpm = None
                    if len(red_signal_per_roi[buffer_key]) >= fps * BUFFER_SECONDS:
                        bpm = calculate_heartbeat(np.array(red_signal_per_roi[buffer_key]), fps)

                    # Desenhar retângulo e informações básicas
                    if laplacian_var < TEXTURE_THRESHOLD or mean_brightness > BRIGHTNESS_THRESHOLD or not motion_detected:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        cv2.putText(frame, "NAO VIVO", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        if use_recognition:
                            # Reconhecimento usando InsightFace
                            name, confidence = recognize_face(frame, face_app, embeddings)
                            
                            if name and confidence > 0.8:  # Ajuste para 80% de certeza
                                confidence_text = "Alta"
                                color = (0, 255, 0)  # Verde
                                cv2.putText(frame, f"{name}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                                cv2.putText(frame, f"Conf: {confidence_text} ({confidence:.2f})", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                            else:
                                cv2.putText(frame, "Desconhecido", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            
                            if bpm is not None:
                                cv2.putText(frame, f"BPM: {int(bpm)}", (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Mostrar status de distância
                    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow("Face Detection", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
            except Exception as e:
                print(f"Erro durante a captura: {e}")
                continue
        
    except Exception as e:
        print(f"Erro ao inicializar captura de tela: {e}")
    
    finally:
        sct.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
