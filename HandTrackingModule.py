import cv2
import mediapipe as mp


class HandDetector:
    def __init__(self,
                 static_image_mode: bool = False,
                 max_num_hands: int = 2,
                 model_complexity: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):

        self.mode = static_image_mode
        self.maxHands = max_num_hands
        self.complexity = model_complexity
        self.detectionConf = min_detection_confidence
        self.trackingConf = min_tracking_confidence

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            model_complexity=self.complexity,
            min_detection_confidence=self.detectionConf,
            min_tracking_confidence=self.trackingConf
        )

    def find_hands(self, img, draw=True, style=False):
        imgRGB = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(image=imgRGB)
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_drawing.draw_landmarks(
                        img,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS)
                if draw and style:
                    self.mp_drawing.draw_landmarks(
                        img,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style())
        return img

    def find_positions(self, img, draw=True, hand_no=0):
        landmarks_list = []
        if self.results.multi_hand_landmarks:
            # It will return a list
            hand_info = self.results.multi_hand_landmarks[hand_no]
            for lm_id, lm in enumerate(hand_info.landmark):
                # print(id, lm)
                # Find out the pixel values
                height, width, channel = img.shape
                channel_x, channel_y = int(lm.x * width), int(lm.y * height)
                # print(id, channel_x, channel_y)
                landmarks_list.append([lm_id, channel_x, channel_y])
                if draw:
                    cv2.circle(img=img, center=(channel_x, channel_y), radius=7, color=(255, 0, 255),
                               thickness=cv2.FILLED)
        return landmarks_list


def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    while True:
        success, img = cap.read()
        if success:
            img = detector.find_hands(img=img)
            landmarks_list = detector.find_positions(img=img, draw=False)
            if len(landmarks_list) != 0:
                print(landmarks_list)
            cv2.imshow("image", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == '__main__':
    main()
