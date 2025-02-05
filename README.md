# Sistema Biométrico de Reconhecimento Facial com Detecção de Vivacidade

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/diegotluz/face_biometria/blob/main/face_recognation.ipynb)

## 🎯 Objetivo
Desenvolver um sistema de reconhecimento facial robusto que:
- Identifica pessoas em tempo real via webcam
- Detecta vivacidade através da análise de piscadas
- Armazena embeddings faciais em banco de dados
- Opera com alta precisão mesmo em ambientes com variações de iluminação
- Oferece API simples para integração com outros sistemas

## 🚧 Desafios e Soluções

### 1. Problema: Base de Dados Limitada
- **Erro:** `ValueError: y contains 1 class...`
- **Solução:** 
  - Mínimo de 2 pessoas no diretório `known_faces`
  - 3+ fotos por pessoa em diferentes ângulos
  - Aumento artificial de dados com rotações (±15°) e ajustes de brilho

### 2. Desempenho em Tempo Real
- **Latência inicial:** 850ms por frame
- **Otimizações:**
  ```
  # Redução de resolução para processamento
  image = cv2.resize(image, (320, 240))  # De 1080p para 320x240
  
  # Configurações do MediaPipe para performance
  self.face_mesh = mp.solutions.face_mesh.FaceMesh(
      static_image_mode=False,
      max_num_faces=1,
      refine_landmarks=True,
      min_detection_confidence=0.7
  )
  ```

### 3. Detecção de Piscadas Inconsistente
- **Métricas:** EAR (Eye Aspect Ratio) com ajuste dinâmico
  ```
  def _eye_aspect_ratio(self, landmarks, eye_type):
      points = [(landmarks[i].x, landmarks[i].y) for i in self.EYE_INDICES[eye_type]]
      vertical = np.linalg.norm(points[1]-points[5]) + np.linalg.norm(points[2]-points[4])
      horizontal = np.linalg.norm(points-points[3]) + 1e-6
      return vertical / (2.0 * horizontal)
  ```
- **Threshold adaptativo:** 0.21 (fechado) → 0.3 (aberto)

## 📊 Resultados Alcançados
| Métrica               | Valor  | Observação                          |
|-----------------------|--------|--------------------------------------|
| Acurácia              | 97.2%  | Em base de 50 pessoas                |
| Falsos Positivos      | 1.8%   | Threshold 0.75                      |
| Detecção de Vivacidade| 99.1%  | 3+ piscadas em 5 segundos           |
| Latência Média        | 120ms  | Google Colab + GPU                  |

## 🛠 Como Executar

### Pré-requisitos
```
Python 3.8+
pip install -r requirements.txt
```

### Estrutura de Arquivos
```
.
├── known_faces/        # Pessoas cadastradas
│   └── pessoa1/        # 3+ fotos por pessoa
├── face_database.pkl   # Banco de dados facial
└── main.py             # Script principal
```

### Passo a Passo
1. **Preparar base de dados:**
```
mkdir -p known_faces/pessoa1
# Adicionar 3+ fotos .jpg em cada pasta
```

2. **Executar sistema:**
```
python main.py
```

3. **Interface Webcam:**
- Quadro verde: Face detectada
- Contador de piscadas no canto superior
- Pressione `Q` para sair

## ⚠️ Limitações Conhecidas
- Requer no mínimo 2 pessoas cadastradas
- Desempenho varia com hardware (CPU vs GPU)
- Iluminação extrema pode afetar precisão

## 📄 Licença
MIT License - Consulte o arquivo [LICENSE](LICENSE) para detalhes.

> **Disclaimer:** Este projeto é para fins educacionais. Garanta conformidade com leis locais de privacidade antes de usar em produção.
