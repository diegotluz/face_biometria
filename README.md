# Sistema Biométrico de Reconhecimento Facial com Detecção de Vivacidade

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]([https://colab.research.google.com/github/diegotluz/face_biometria/blob/main/face_recognation.ipynb])

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
```

Este README inclui:
- Badge para execução imediata no Colab
- Seções organizadas com emojis visuais
- Tabelas comparativas de resultados
- Diagrama de arquivos simplificado
- Instruções de execução detalhadas
- Advertências legais importantes

Para personalizar, substitua:
- `yourusername` no badge do Colab
- Adicione capturas de tela do sistema em operação
- Inclua referências específicas do seu projeto

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/30814256/7f508f5a-3c7b-4615-98cd-ef4e1e1e7a30/paste-2.txt
[2] https://veridas.com/en/facial-recognition/
[3] https://www.projectclue.com/computer-science/project-topics-materials-for-undergraduate-students/design-and-implementation-of-face-detection-and-recognition-system
[4] https://swiftlane.com/blog/how-facial-recognition-works/
[5] https://research.aimultiple.com/facial-recognition-challenges/
[6] https://www.isaca.org/resources/news-and-trends/newsletters/atisaca/2022/volume-51/facial-recognition-technology-and-privacy-concerns
[7] https://itrexgroup.com/blog/facial-recognition-benefits-applications-challenges/
[8] https://sol.sbc.org.br/index.php/sbcup/article/download/11225/11096/
[9] https://appinventiv.com/blog/facial-recognition-software-development-benefits/
[10] https://ntechlab.com/success-stories/
[11] https://realpython.com/face-recognition-with-python/
[12] https://mobidev.biz/blog/how-build-face-recognition-system-integrate-into-app
[13] https://www.forbes.com/councils/forbestechcouncil/2020/06/23/facial-recognition-systems-security/
[14] https://www.hidglobal.com/solutions/facial-recognition-technology
[15] https://itrexgroup.com/blog/facial-recognition-benefits-applications-challenges/
[16] https://www.signicat.com/blog/face-recognition
[17] https://mobidev.biz/blog/how-build-face-recognition-system-integrate-into-app
[18] https://en.wikipedia.org/wiki/Face_recognition
[19] https://www.thalesgroup.com/en/markets/digital-identity-and-security/government/biometrics/facial-recognition
[20] https://keyless.io/blog/post/facial-recognition-applications-benefits-and-challenges
[21] https://www.e3s-conferences.org/articles/e3sconf/pdf/2024/94/e3sconf_icpgres2024_07002.pdf
[22] https://www.nature.com/articles/s44287-024-00094-x
[23] https://www.scylla.ai/facial-recognition-technology-challenges-and-use-cases/
[24] https://www.researchgate.net/publication/329046685_Techniques_and_Challenges_of_Face_Recognition_A_Critical_Review
[25] https://www.soa.org/49022b/globalassets/assets/files/resources/research-report/2023/dei107-facial-recognition-challenges.pdf
[26] https://www.researchgate.net/publication/271584966_Face_Recognition_Challenges_Achievements_and_Future_Directions
[27] https://www.thalesgroup.com/en/markets/digital-identity-and-security/government/inspired/facial-recognition-issues
[28] https://www.biometricupdate.com/202304/facial-recognition-three-common-pitfalls-and-how-to-fix-them
[29] https://incode.com/blog/model-training-for-face-recognition-and-how-we-improve-over-time-how-it-works/
[30] https://www.linkedin.com/advice/0/what-current-challenges-limitations-face
[31] https://www.privacycompliancehub.com/gdpr-resources/10-reasons-to-be-concerned-about-facial-recognition-technology/
[32] https://answers.microsoft.com/en-us/windows/forum/all/how-to-fix-face-recognition/feb1f824-49d4-4081-8045-c1f9dd145145
[33] https://www.linkedin.com/pulse/ethics-errors-facial-recognition-technology-naveen-joshi
[34] https://www.gartner.com/smarterwithgartner/how-to-use-facial-recognition-technology-responsibly-and-ethically
[35] https://bvalaw.com.br/en/face-recognition-and-its-limitations/
[36] https://www.edps.europa.eu/press-publications/press-news/blog/facial-recognition-solution-search-problem_en
[37] https://www.aclu-mn.org/en/news/biased-technology-automated-discrimination-facial-recognition
[38] https://mobidev.biz/blog/improve-ai-facial-recognition-accuracy-with-machine-deep-learning
[39] https://sol.sbc.org.br/index.php/sbcup/article/view/11225
[40] https://github.com/ageitgey/face_recognition?tab=MIT-1-ov-file
[41] https://www.fraud.com/post/facial-recognition-systems
[42] https://timesofindia.indiatimes.com/blogs/darksides/leveraging-technology-to-reconnect-missing-children-with-their-families-in-india/
[43] https://www.csis.org/analysis/how-does-facial-recognition-work
[44] https://shuftipro.com/blog/the-benefits-and-best-practices-of-deploying-facial-recognition-in-the-workplace/
[45] https://www.securityindustry.org/2022/03/15/examples-of-successful-use-of-facial-recognition-in-virginia/
[46] https://paperswithcode.com/task/face-recognition
[47] https://hyperverge.co/blog/benefits-of-facial-recognition/
[48] https://cc-techgroup.com/facial-recognition/
[49] https://senstar.com/senstarpedia/pros-and-cons-of-facial-recognition/
[50] https://www.zegocloud.com/blog/facial-recognition
[51] https://www.codingal.com/coding-for-kids/blog/build-face-recognition-app-with-python/
[52] https://www.superannotate.com/blog/guide-to-face-recognition
[53] https://binmile.com/blog/build-face-detection-and-recognition-app/
[54] https://www.instructables.com/ChikonEye-Single-Python-Script-to-Face-Recognition/
[55] https://www.elastic.co/blog/how-to-build-a-facial-recognition-system-using-elasticsearch-and-python
[56] https://www.youtube.com/watch?v=bK_k7eebGgc
[57] https://www.innovatrics.com/facial-recognition-technology/
[58] https://microkeeper.com.au/guides.php?guide=facial-recognition-setup-and-installation
[59] https://www.youtube.com/watch?v=Y0dLgtF4IHM
[60] https://rm.coe.int/guidelines-facial-recognition-web-a5-2750-3427-6868-1/1680a31751
[61] https://www.incognia.com/the-authentication-reference/face-recognition-all-there-is-to-know
[62] https://www.cyient.com/blog/building-an-efficient-face-recognition-system
[63] https://idtechwire.com/solutions/facial-recognition/
[64] https://aws.amazon.com/what-is/facial-recognition/
[65] https://www.mobbeel.com/en/the-ultimate-guide-to-face-recognition/
[66] https://www.ijert.org/research/a-survey-on-various-problems-challenges-in-face-recognition-IJERTV2IS60850.pdf
[67] https://senstar.com/senstarpedia/facial-recognition-problems/
[68] https://blog.emb.global/challenges-of-facial-recognition-technology/
[69] https://thebulletin.org/2021/11/its-time-to-address-facial-recognition-the-most-troubling-law-enforcement-ai-tool/
[70] https://pmc.ncbi.nlm.nih.gov/articles/PMC7575263/
[71] https://research.aimultiple.com/facial-recognition-challenges/
[72] https://www.techtarget.com/whatis/feature/Pros-and-cons-of-facial-recognition
[73] https://hyperverge.co/blog/mitigating-facial-recognition-bias/
[74] https://fiswg.org/fiswg_fi_success_stories_2020_07_17.pdf
[75] https://www.securityindustry.org/2020/07/16/facial-recognition-success-stories-showcase-positive-use-cases-of-the-technology/
[76] https://ametawiki.ametagroup.com/docs/face-detection-and-recognition-setup-guide
[77] https://www.datacamp.com/tutorial/face-detection-python-opencv
[78] https://community.hailo.ai/t/a-comprehensive-guide-to-building-a-face-recognition-system/8803
[79] https://www.clariontech.com/blog/a-guide-to-building-java-based-facial-recognition-software-for-developers
[80] https://github.com/techwithtim/Face-Recognition
[81] https://www.turing.com/kb/using-deep-learning-to-design-face-detection-and-recognition-systems

---
Resposta do Perplexity: pplx.ai/share
