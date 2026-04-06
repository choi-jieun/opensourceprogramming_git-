print("코드 시작됨")

import cv2

def detect_faces_from_webcam():
    # OpenCV에서 제공하는 Haar Cascade 얼굴 검출기 로드
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    if face_cascade.empty():
        print("얼굴 검출기 로드 실패")
        return

    # 웹캠 열기 (기본 카메라: 0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("웹캠을 열 수 없음")
        return

    print("실행 중... 종료하려면 'q'를 누르셈.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없음")
            break

        # 흑백 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 얼굴 검출
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        # 검출된 얼굴에 사각형 표시
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 얼굴 수 표시
        cv2.putText(
            frame,
            f"Faces detected: {len(faces)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Face Detection", frame)

        # q 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_faces_from_webcam()