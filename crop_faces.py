import cv2
import os
import numpy as np
from pathlib import Path

def create_augmented_face(face_img, angle=None, scale=None):
    """Aplica augmentation na imagem do rosto"""
    height, width = face_img.shape[:2]
    
    if angle is not None:
        # Rotação
        center = (width//2, height//2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        face_img = cv2.warpAffine(face_img, rotation_matrix, (width, height))
    
    if scale is not None:
        # Redimensionamento
        face_img = cv2.resize(face_img, None, fx=scale, fy=scale)
        
    # Ajusta para o tamanho final desejado (250x250)
    face_img = cv2.resize(face_img, (250, 250))
    
    return face_img

def process_images(input_dir, output_base_dir, target_size=(250, 250)):
    # Carrega o classificador de rosto
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
    
    # Para cada pasta de pessoa no diretório de entrada
    for person_dir in Path(input_dir).iterdir():
        if not person_dir.is_dir():
            continue
            
        person_name = person_dir.name
        output_dir = Path(output_base_dir) / person_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Processa cada imagem da pessoa
        for img_path in person_dir.glob('*.jpg'):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for i, (x, y, w, h) in enumerate(faces):
                # Adiciona margem ao rosto detectado
                margin = int(0.2 * w)  # 20% de margem
                x = max(0, x - margin)
                y = max(0, y - margin)
                w = min(img.shape[1] - x, w + 2*margin)
                h = min(img.shape[0] - y, h + 2*margin)
                
                face = img[y:y+h, x:x+w]
                
                # Gera variações do rosto
                augmentations = [
                    # Original
                    (None, None),
                    # Rotações
                    (-15, None), (15, None),
                    # Escalas
                    (None, 0.9), (None, 1.1),
                    # Combinações
                    (-15, 0.9), (15, 1.1)
                ]
                
                for j, (angle, scale) in enumerate(augmentations):
                    aug_face = create_augmented_face(face, angle, scale)
                    
                    # Define nome do arquivo
                    aug_type = f"_rot{angle}" if angle else ""
                    aug_type += f"_scale{scale}" if scale else ""
                    if not aug_type:
                        aug_type = "_original"
                    
                    output_filename = f"{img_path.stem}_face{i}{aug_type}.jpg"
                    output_path = output_dir / output_filename
                    
                    cv2.imwrite(str(output_path), aug_face)

if __name__ == "__main__":
    input_dir = 'known_faces'  # Estrutura: known_faces/nome_pessoa/imagens.jpg
    output_dir = 'processed_faces'
    
    process_images(input_dir, output_dir)
    print("Processamento concluído. Dataset aumentado salvo em 'processed_faces'.") 