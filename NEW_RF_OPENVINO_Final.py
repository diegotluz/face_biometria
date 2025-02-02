import os,time
import cv2
import mediapipe as mp
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks, butter, filtfilt


# Suprimir avisos
os.environ['TF_ENABLE_ONEDNN_OPTS']= '5'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
import logging
logging.getLogger('mediapipe').setLevel(logging.ERROR)

# Configurações de performance
cv2.ocl.setUseOpenCL(True)
cv2.setNumThreads(10)

# Configurações
KNOWN_FACES_PATH = "./known_faces"
DATABASE_PATH = "face_database.pkl"
FPS = 30
BPM_WINDOW = 150
DEPTH_HISTORY = 10
WINDOW_NAME = "Sistema Biométrico Avançado"

class FaceBiometricSystem:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.database = self.load_or_create_database()
        self.scaler = StandardScaler()
        self.clf = self.train_model()
        
        self.depth_buffer = []
        self.bpm_signal = []
        self.last_bpm = None
        self.eye_blink_counter = 0
        self.last_blink_time = time.time()
        self.head_movement_buffer = []

    def detect_liveness(self, landmarks):
        # 1. Detecção de piscar de olhos
        left_eye = self.eye_aspect_ratio(landmarks[159], landmarks[145], landmarks[133])  # OLHO ESQUERDO
        right_eye = self.eye_aspect_ratio(landmarks[386], landmarks[374], landmarks[362])  # OLHO DIREITO
        ear = (left_eye + right_eye) / 2.0
        
        # Detecta piscar (threshold ajustável)
        if ear < 0.21:
            if time.time() - self.last_blink_time > 0.3:  # Evita detecções múltiplas
                self.eye_blink_counter += 1
                self.last_blink_time = time.time()

        # 2. Análise de movimento da cabeça
        nose_tip = np.array([landmarks[4].x, landmarks[4].y])
        self.head_movement_buffer.append(nose_tip)
        if len(self.head_movement_buffer) > 30:  # 1 segundo de histórico
            self.head_movement_buffer.pop(0)
        
        # 3. Análise de profundidade 3D
        depth = self.calculate_depth(landmarks)
        
        # 4. BPM fisiológico
        bpm = self.last_bpm if self.last_bpm else 0
        
        # Combinação de fatores
        liveness_score = 0
        liveness_score += min(self.eye_blink_counter, 2) * 25  # Máx 50 pontos
        liveness_score += min(np.std(self.head_movement_buffer)*1000, 25)  # Máx 25 pontos
        liveness_score += min((bpm/2) if 60 < bpm < 100 else 0, 25)  # Máx 25 pontos
        
        return liveness_score >= 50  # Threshold para considerar vivo

    def eye_aspect_ratio(self, eye_top, eye_bottom, eye_outer):
        # Calcula a relação de aspecto do olho (EAR)
        vertical_dist = np.linalg.norm(np.array([eye_top.x, eye_top.y]) - 
                                     np.array([eye_bottom.x, eye_bottom.y]))
        horizontal_dist = np.linalg.norm(np.array([eye_outer.x, eye_outer.y]) - 
                                        np.array([eye_outer.x, eye_outer.y]))
        return vertical_dist / (horizontal_dist + 1e-6)

    def load_or_create_database(self):
        try:
            with open(DATABASE_PATH, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return self.build_database()

    def build_database(self):
        database = {}
        for person in os.listdir(KNOWN_FACES_PATH):
            person_dir = os.path.join(KNOWN_FACES_PATH, person)
            if os.path.isdir(person_dir):
                embeddings = []
                for img_file in os.listdir(person_dir):
                    if img_file.lower().endswith(('png', 'jpg', 'jpeg')):
                        img_path = os.path.join(person_dir, img_file)
                        img = cv2.imread(img_path)
                        if img is not None:
                            embedding = self.extract_embedding(img)
                            if embedding is not None:
                                embeddings.append(embedding)
                if embeddings:
                    database[person] = {
                        'embeddings': embeddings,
                        'avg_embedding': np.mean(embeddings, axis=0),
                        'samples': len(embeddings)
                    }
        self.save_database(database)
        return database

    def extract_embedding(self, image):
        results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            return self.process_landmarks(results.multi_face_landmarks[0].landmark)
        return None

    def process_landmarks(self, landmarks):
        key_points = [33, 133, 362, 263, 61, 291, 4, 164, 0, 13, 14, 17, 291, 37]
        return np.array([(landmarks[i].x, landmarks[i].y, landmarks[i].z) for i in key_points]).flatten()

    def train_model(self):
        X = []
        y = []
        
        persons = list(self.database.keys())
        for person in persons:
            embeddings = self.database[person]['embeddings']
            
            for i in range(len(embeddings)):
                for j in range(i+1, len(embeddings)):
                    features = self.hybrid_metric(embeddings[i], embeddings[j])
                    X.append(features)
                    y.append(1)
            
            for other in [p for p in persons if p != person]:
                other_emb = self.database[other]['embeddings'][0]
                for emb in embeddings:
                    features = self.hybrid_metric(emb, other_emb)
                    X.append(features)
                    y.append(0)

        X = np.array(X)
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        clf = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
            subsample=0.8
        )
        clf.fit(X_scaled, y)
        return clf

    def hybrid_metric(self, emb1, emb2):
        l1 = np.linalg.norm(emb1 - emb2, 1)
        l2 = np.linalg.norm(emb1 - emb2)
        cos = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)
        canb = np.sum(np.abs(emb1 - emb2) / (np.abs(emb1) + np.abs(emb2) + 1e-8))
        return np.array([l1, l2, cos, canb])

    def save_database(self, database):
        with open(DATABASE_PATH, 'wb') as f:
            pickle.dump(database, f)

    def authenticate(self, frame):
        results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return "Desconhecido", 0.0, 0.0, None, False

        landmarks = results.multi_face_landmarks[0].landmark
        emb = self.process_landmarks(landmarks)
        if emb is None:
            return "Desconhecido", 0.0, 0.0, None, False

        # Cálculos biométricos
        depth = self.calculate_depth(landmarks)
        face_roi = self.get_face_roi(frame, landmarks)
        bpm = self.calculate_bpm(face_roi)
        liveness = self.detect_liveness(landmarks)

        # Reconhecimento
        best_match = "Desconhecido"
        best_score = 0.0
        for person in self.database:
            for db_emb in self.database[person]['embeddings']:
                features = self.hybrid_metric(emb, db_emb)
                features = self.scaler.transform([features])
                score = self.clf.predict_proba(features)[0][1]
                
                if score > best_score and score > 0.7:
                    best_match = person
                    best_score = score

        return best_match, best_score, depth, bpm, liveness

    def calculate_depth(self, landmarks):
        left_eye = np.array([landmarks[33].x, landmarks[33].y, landmarks[33].z])
        right_eye = np.array([landmarks[263].x, landmarks[263].y, landmarks[263].z])
        depth = np.linalg.norm(left_eye - right_eye)
        self.depth_buffer = ([depth] + self.depth_buffer)[:DEPTH_HISTORY]
        return np.median(self.depth_buffer)

    def get_face_roi(self, frame, landmarks):
        x_coords = [int(l.x * frame.shape[1]) for l in landmarks]
        y_coords = [int(l.y * frame.shape[0]) for l in landmarks]
        return frame[min(y_coords):max(y_coords), min(x_coords):max(x_coords)]

    def calculate_bpm(self, face_roi):
        if face_roi.size == 0:
            return None
            
        green = np.mean(face_roi[:,:,1])
        self.bpm_signal.append(green)
        
        if len(self.bpm_signal) >= BPM_WINDOW:
            signal = np.array(self.bpm_signal)
            signal -= np.mean(signal)
            
            b, a = butter(2, [0.7/(FPS/2), 4/(FPS/2)], btype='band')
            filtered = filtfilt(b, a, signal)
            
            peaks, _ = find_peaks(filtered, distance=FPS//2, prominence=0.5)
            
            if len(peaks) >= 2:
                self.last_bpm = int(60 * FPS / np.mean(np.diff(peaks)))
                self.bpm_signal = []
        
        return self.last_bpm

    def run(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Erro ao abrir a câmera!")
            return

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)
                name, score, depth, bpm, liveness = self.authenticate(frame)
                
                # Exibição dos resultados
                color = (0, 255, 0) if score > 0.7 else (0, 0, 255)
                status = f"{name} ({score:.2f})" if score > 0.5 else "Desconhecido"
                liveness_status = "Vivo: Sim" if liveness else "Vivo: Não"
                liveness_color = (0, 255, 0) if liveness else (0, 0, 255)
                
                cv2.putText(frame, status, (20, 40), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(frame, f"Profundidade: {depth:.2f}mm", (20, 80),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.putText(frame, f"BPM: {bpm or '--'}", (20, 120),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.putText(frame, liveness_status, (20, 160),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, liveness_color, 2)

                cv2.imshow(WINDOW_NAME, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.face_mesh.close()

if __name__ == "__main__":
    print("Inicializando sistema biométrico...")
    system = FaceBiometricSystem()
    print("Base de dados carregada com sucesso!")
    print("Resumo do banco de dados:")
    for person, data in system.database.items():
        print(data)
        # print(f" - {person}: {data['samples']} amostras")
    system.run()
