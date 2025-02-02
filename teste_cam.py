import cv2

cap = cv2.VideoCapture(2)  # Use 0 para a webcam padrão

if not cap.isOpened():
    print("Erro: Não foi possível acessar a webcam.")
else:
    print("Webcam acessada com sucesso. Exibindo vídeo...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro: Não foi possível capturar o quadro da webcam.")
            break

        cv2.imshow("Teste de Webcam", frame)

        # Pressione 'q' para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
