import cv2

from deepface import DeepFace
import time


def main():
    # Carrega o classificador Haar cascade para detecção de faces
    # Você pode precisar baixar o arquivo XML, se não for encontrado automaticamente
    # O DeepFace geralmente lida com a detecção, mas o OpenCV é usado para captura e exibição
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Inicia a captura de vídeo da webcam (0 para a câmera padrão)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error openning webcam.")
        exit()

    # Variável para controlar a frequência de detecção de emoções
    # A detecção de emoções é mais lenta que a detecção de faces, então não a rodaremos em todos os frames
    last_detection_time = time.time()
    results_text = "Detecting..."

    while True:
        # Captura frame por frame
        ret, frame = cap.read()
        if not ret:
            break

        # Converte o frame para escala de cinza para a detecção de faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detecta faces no frame
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        # Processa cada face detectada
        for x, y, w, h in faces:
            # Desenha um retângulo ao redor do rosto
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Realiza a análise de emoção em um intervalo regular (ex: a cada 2 segundos)
            if time.time() - last_detection_time > 2:
                try:
                    # Extrai a região de interesse (ROI) do rosto para análise
                    face_roi = frame[y : y + h, x : x + w]
                    # Analisa a emoção usando DeepFace
                    result = DeepFace.analyze(
                        face_roi,
                        actions=["emotion", "gender", "age", "race"],
                        enforce_detection=False,
                    )
                    if isinstance(result, list):
                        result = result[0]

                    # Pega a emoção dominante
                    emotion = result["dominant_emotion"]
                    age = result["age"]
                    gender = result["dominant_gender"]
                    dominant_race = result["dominant_race"]

                    results_text = f"{gender}, {age}y, {dominant_race}, {emotion}"

                    last_detection_time = time.time()
                except Exception as e:
                    results_text = "Not detected"

            # Coloca o texto da emoção no frame
            # cv2.putText(frame, results_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(
                frame,
                results_text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        # Exibe o frame resultante
        cv2.imshow("Facial Emotion Detection", frame)

        # Pressionar 'q' para sair do loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Libera a captura de vídeo e fecha todas as janelas
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
