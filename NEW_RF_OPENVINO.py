import os
import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
from collections import deque
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Configurações do sistema
KNOWN_FACES_PATH = "./known_faces"
DATABASE_PATH = "face_database.pkl"
FPS = 30
BLINK_THRESHOLD = 0.21
CHALLENGE_BLINKS = 3
SIMILARITY_THRESHOLD = 0.82
EAR_CONSEC_FRAMES = 2
FACE_TIMEOUT = 7  # segundos sem detecção de rosto
WINDOW_NAME = "BioAuth System v2.0"

# Inicialização do MediaPipe
mp_face_mesh = mp.solutions.face_mesh

class BioAuthSystem:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.72,
            min_tracking_confidence=0.62
        )
        
        # Carregamento do banco de dados
        self.database = self._load_database()
        self.scaler = StandardScaler()
        self._train_model()
        
        # Estado do sistema
        self.eye_counter = 0
        self.eye_frames = 0
        self.challenge_complete = False
        self.last_face_time = time.time()
        self.last_blink_time = time.time()
        
        # Buffers para suavização
        self.EAR_HISTORY = deque(maxlen=15)
        self.embedding_buffer = deque(maxlen=12)
        
        # Pontos dos olhos otimizados
        self.EYE_INDICES = {
            'left': [33, 160, 159, 158, 133, 153, 145],
            'right': [362, 385, 386, 387, 263, 373, 374]
        }

    def _load_database(self):
        try:
            with open(DATABASE_PATH, 'rb') as f:
                return pickle.load(f)
        except (FileNotFoundError, EOFError):
            return self._build_database()

    def _build_database(self):
        database = {}
        for person in os.listdir(KNOWN_FACES_PATH):
            person_dir = os.path.join(KNOWN_FACES_PATH, person)
            if os.path.isdir(person_dir):
                embeddings = []
                for img_file in os.listdir(person_dir):
                    if img_file.lower().endswith(('png', 'jpg', 'jpeg')):
                        img = cv2.imread(os.path.join(person_dir, img_file))
                        if img is not None:
                            emb = self._get_embedding(img)
                            if emb is not None:
                                embeddings.append(emb)
                if embeddings:
                    database[person] = np.mean(embeddings, axis=0)
        self._save_database(database)
        return database

    def _save_database(self, database):
        with open(DATABASE_PATH, 'wb') as f:
            pickle.dump(database, f)

    def _get_embedding(self, image):
        results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None
            
        landmarks = results.multi_face_landmarks[0].landmark
        key_points = np.array([(landmarks[i].x, landmarks[i].y) for i in [33, 133, 263, 362, 61, 291, 4]])
        return key_points.flatten()

    def _train_model(self):
        if not self.database:
            return
            
        X = list(self.database.values())
        y = list(self.database.keys())
        X = self.scaler.fit_transform(X)
        self.clf = KNeighborsClassifier(n_neighbors=3, metric='cosine')
        self.clf.fit(X, y)

    def _calculate_ear(self, landmarks):
        def eye_aspect_ratio(indices):
            vertical = np.linalg.norm(landmarks[indices[1]] - landmarks[indices[4]]) + \
                      np.linalg.norm(landmarks[indices[2]] - landmarks[indices[5]])
            horizontal = np.linalg.norm(landmarks[indices[0]] - landmarks[indices[3]])
            return vertical / (2.0 * horizontal + 1e-6)
        
        left_ear = eye_aspect_ratio(self.EYE_INDICES['left'])
        right_ear = eye_aspect_ratio(self.EYE_INDICES['right'])
        return (left_ear + right_ear) / 2.0

    def _check_challenge(self, landmarks):
        current_time = time.time()
        ear = self._calculate_ear(landmarks)
        self.EAR_HISTORY.append(ear)
        smoothed_ear = np.mean(self.EAR_HISTORY)
        
        # Detecção de piscada
        if smoothed_ear < BLINK_THRESHOLD:
            self.eye_frames += 1
        else:
            if self.eye_frames >= EAR_CONSEC_FRAMES:
                if (current_time - self.last_blink_time) > 0.4:
                    self.eye_counter = min(self.eye_counter + 1, CHALLENGE_BLINKS)
                    self.last_blink_time = current_time
            self.eye_frames = 0
        
        # Timeout por ausência de rosto
        if (current_time - self.last_face_time) > FACE_TIMEOUT:
            self._reset_challenge()
            
        self.challenge_complete = self.eye_counter >= CHALLENGE_BLINKS
        return self.challenge_complete

    def _reset_challenge(self):
        self.eye_counter = 0
        self.eye_frames = 0
        self.challenge_complete = False
        self.EAR_HISTORY.clear()
        self.embedding_buffer.clear()

    def _process_frame(self, frame):
        results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            self.last_face_time = time.time()
            return frame, None, 0.0, False
            
        landmarks = results.multi_face_landmarks[0].landmark
        self.last_face_time = time.time()
        
        # Processamento do desafio
        if not self.challenge_complete:
            challenge_done = self._check_challenge(landmarks)
            if not challenge_done:
                return self._draw_challenge_ui(frame, landmarks)
        
        # Processamento da autenticação
        return self._process_authentication(frame, landmarks)

    def _process_authentication(self, frame, landmarks):
        embedding = self._get_embedding_from_landmarks(landmarks)
        self.embedding_buffer.append(embedding)
        
        if len(self.embedding_buffer) < 8:
            return self._draw_processing_ui(frame, len(self.embedding_buffer))
            
        avg_embedding = np.mean(self.embedding_buffer, axis=0)
        emb_scaled = self.scaler.transform([avg_embedding])
        
        similarities = cosine_similarity(emb_scaled, list(self.database.values()))
        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[0][best_match_idx]
        
        if best_similarity > SIMILARITY_THRESHOLD:
            identity = list(self.database.keys())[best_match_idx]
            return self._draw_auth_success(frame, identity, best_similarity)
        return self._draw_auth_failure(frame, best_similarity)

    def _get_embedding_from_landmarks(self, landmarks):
        key_points = np.array([(landmarks[i].x, landmarks[i].y) for i in [33, 133, 263, 362, 61, 291, 4]])
        return key_points.flatten()

    def _draw_challenge_ui(self, frame, landmarks):
        h, w = frame.shape[:2]
        
        # Desenho dos pontos dos olhos
        for idx in self.EYE_INDICES['left'] + self.EYE_INDICES['right']:
            x = int(landmarks[idx].x * w)
            y = int(landmarks[idx].y * h)
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        
        # Interface de desafio
        cv2.putText(frame, f"PISQUE {CHALLENGE_BLINKS} VEZES", (20, 40), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f"Piscadas: {self.eye_counter}/{CHALLENGE_BLINKS}", (20, 80),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Barra de progresso
        cv2.rectangle(frame, (20, h-50), (int(w*(self.eye_counter/CHALLENGE_BLINKS)), h-30), 
                    (0, 255, 0), -1)
        
        return frame, None, 0.0, False

    def _draw_processing_ui(self, frame, buffer_count):
        h, w = frame.shape[:2]
        cv2.putText(frame, "ANALISANDO...", (w//2-150, h//2), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"{buffer_count}/{len(self.embedding_buffer)}", 
                  (w//2-30, h//2+40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        return frame, None, 0.0, False

    def _draw_auth_success(self, frame, identity, similarity):
        color = (0, 255, 0)
        cv2.putText(frame, f"AUTENTICADO: {identity}", (20, 40), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Similaridade: {similarity:.2%}", (20, 80),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame, identity, similarity, True

    def _draw_auth_failure(self, frame, similarity):
        color = (0, 0, 255)
        cv2.putText(frame, "NAO AUTENTICADO", (20, 40), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Similaridade: {similarity:.2%}", (20, 80),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame, "Desconhecido", similarity, False

    def run(self):
        self.cap = cv2.VideoCapture(0)
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                processed_frame, identity, similarity, auth_done = self._process_frame(frame)
                
                cv2.imshow(WINDOW_NAME, processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Inicializando sistema biométrico...")
    system = BioAuthSystem()
    system.run()
