import os
import requests
import cv2
import numpy as np
import json
import re

class ImageDownloader:
    def __init__(self):
        self.known_faces_dir = "known_faces"
        
        # Criar diretório se não existir
        if not os.path.exists(self.known_faces_dir):
            os.makedirs(self.known_faces_dir)

    def search_and_download(self, query, num_images=5):
        """
        Busca e baixa imagens do DuckDuckGo usando a query fornecida
        """
        try:
            # Preparar a URL da API do DuckDuckGo
            url = "https://duckduckgo.com/"
            params = {
                'q': query + " face",
                'iax': 'images',
                'ia': 'images'
            }
            
            # Headers para simular um navegador
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            # Primeira requisição para obter o token vqd
            res = requests.post(url, data=params, headers=headers)
            search_obj = re.search(r'vqd=([\d-]+)\&', res.text, re.M|re.I)
            if not search_obj:
                print("Token vqd não encontrado")
                return False
            
            vqd = search_obj.group(1)
            
            # URL para obter os resultados das imagens
            url = f"https://duckduckgo.com/i.js"
            params = {
                'l': 'br-pt',
                'o': 'json',
                'q': query + " face",
                'vqd': vqd,
                'f': ',,,',
                'p': '1',
                'v7exp': 'a'
            }
            
            # Fazer a requisição para obter as imagens
            res = requests.get(url, headers=headers, params=params)
            data = json.loads(res.text)
            
            if 'results' not in data:
                print(f"Nenhuma imagem encontrada para: {query}")
                return False

            # Limitar ao número de imagens solicitado
            image_urls = [img['image'] for img in data['results'][:num_images]]
            
            # Processar cada imagem encontrada
            success_count = 0
            for idx, image_url in enumerate(image_urls):
                try:
                    # Baixar imagem
                    response = requests.get(image_url, headers=headers, timeout=10)
                    if response.status_code == 200:
                        # Converter para formato OpenCV
                        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                        
                        if image is not None:
                            # Detectar face
                            face_cascade = cv2.CascadeClassifier(
                                cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
                            )
                            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                            
                            if len(faces) > 0:
                                # Salvar imagem se contiver face
                                filename = f"{query}_{idx+1}.jpg"
                                filepath = os.path.join(self.known_faces_dir, filename)
                                cv2.imwrite(filepath, image)
                                success_count += 1
                                print(f"Imagem salva: {filename}")
                            else:
                                print(f"Nenhuma face detectada na imagem {idx+1}")
                                
                except Exception as e:
                    print(f"Erro ao processar imagem {idx+1}: {str(e)}")
                    continue

            print(f"\nTotal de imagens baixadas com sucesso: {success_count}")
            return success_count > 0

        except Exception as e:
            print(f"Erro durante a busca: {str(e)}")
            return False

    def manual_download(self, image_url, person_name):
        """
        Baixa uma imagem específica a partir da URL fornecida
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(image_url, headers=headers, timeout=10)
            if response.status_code == 200:
                # Converter para formato OpenCV
                image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                
                if image is not None:
                    # Detectar face
                    face_cascade = cv2.CascadeClassifier(
                        cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
                    )
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                    
                    if len(faces) > 0:
                        filename = f"{person_name}.jpg"
                        filepath = os.path.join(self.known_faces_dir, filename)
                        cv2.imwrite(filepath, image)
                        print(f"Imagem salva com sucesso: {filename}")
                        return True
                    else:
                        print("Nenhuma face detectada na imagem")
                        return False
            else:
                print(f"Erro ao baixar imagem. Status code: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"Erro durante o download: {str(e)}")
            return False

def main():
    downloader = ImageDownloader()
    
    while True:
        print("\n=== Menu de Download de Imagens ===")
        print("1. Buscar e baixar imagens do DuckDuckGo")
        print("2. Baixar imagem específica por URL")
        print("3. Sair")
        
        opcao = input("\nEscolha uma opção: ")
        
        if opcao == "1":
            query = input("Digite o nome da pessoa para buscar: ")
            num_images = int(input("Quantas imagens deseja baixar (máx. 50)? "))
            num_images = min(50, max(1, num_images))  # Limitar entre 1 e 50
            downloader.search_and_download(query, num_images)
            
        elif opcao == "2":
            url = input("Digite a URL da imagem: ")
            nome = input("Digite o nome da pessoa: ")
            downloader.manual_download(url, nome)
            
        elif opcao == "3":
            print("Encerrando programa...")
            break
            
        else:
            print("Opção inválida!")

if __name__ == "__main__":
    main()
