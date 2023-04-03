# Find the information of hands
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

hands = mp_hands.Hands()

# For webcam input:
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if success:
        imgRGB = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)
        results = hands.process(image=imgRGB)
        # it will return a list
        info = results.multi_hand_landmarks
        if info is not None:
            print(info[0])
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # print(hand_landmarks)
                for id, lm in enumerate(hand_landmarks.landmark):
                    # print(id, lm)
                    # Find out the pixel values
                    height, width, channel = img.shape
                    channel_x, channel_y = int(lm.x * width), int(lm.y * height)
                    # print(id, channel_x, channel_y)
                    # if id == 0:
                    #     cv2.circle(img=img, center=(channel_x, channel_y), radius=15, color=(255, 0, 255),
                    #                thickness=cv2.FILLED)
                mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    # mp_drawing_styles.get_default_hand_landmarks_style(),
                    # mp_drawing_styles.get_default_hand_connections_style()
                )
        cv2.imshow("image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
