import cv2
import mediapipe as mp

# 손 감지 모듈 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.1)

# 화면 캡처 장치 초기화
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue
    
    # 카메라 화면 좌우 반전
    frame = cv2.flip(frame, 1)
    
    # 손 감지 수행
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # 감지된 손이 있을 경우에만 손가락 추적
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 각 손가락의 좌표 추출하여 관절 부분 표시
            for finger_landmark in hand_landmarks.landmark:
                x = int(finger_landmark.x * frame.shape[1])
                y = int(finger_landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    
    # 손이 감지되지 않은 경우, 화면에 메시지 표시
    else:
        cv2.putText(frame, "No hand detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Finger Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
