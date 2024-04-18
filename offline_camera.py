import cv2
from app import Detectmorse


def main():
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(r"C:\Users\HP\Desktop\story\morse_code-eyeblink-to-text\blinking.mp4")
    camera = Detectmorse()
    
    success = True
    while success:
        try:
            success, frame = cap.read()
            # if not success:
            #     # print("Ignoring empty camera frame.")
            #     # print("Try again")
            #     break

            # Process frame and display it
            text, frame = camera.calculate(frame)  # Capture returned frame
            
            cv2.imshow("Frame", frame)

            # cv2.putText(frame, "Predicted :  " + text, (10, 470),
            #             cv2.FONT_HERSHEY_DUPLEX, 0.7, (52, 152, 219), 2)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            
        except Exception as e: 
            print(f"An error occurred: {e}")
            break
    cap.release()
    cv2.destroyAllWindows()
    print(text)


if __name__ == '__main__':
    main()
