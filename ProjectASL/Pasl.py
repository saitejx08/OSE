# from ultralytics import YOLO
# import cv2
# import os
# import datetime
#
# SCREE_SIZE = (1280,720)
# model = YOLO(r"D:\PythonProject\ProjectASL\hand.pt")
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREE_SIZE[0])
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREE_SIZE[1])
#
# recognized_text=""
# current_letter= None
# stable_letter = None
# hand_present = False
#
# letter_buffer =[]
# BUFFER_SIZE = 10
#
# os.makedirs("snapshots", exist_ok=True)
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     results= model.predict(source=frame, conf=0.5,verbose=False)
#     annotated_frame = results[0].plot()
#
#     if results[0].boxes:
#         class_id = int(results[0].boxes.cls[0])
#         letter_buffer.append(model.names[class_id])
#
#         if len(letter_buffer) > BUFFER_SIZE:
#                 letter_buffer.pop(0)
#
#             else:
#                 if hand_present and stable_letter:
#                     recognized_text += stable_letter
#                     print(f"finalized letter: {stable_letter}")
#                     stable_letter = None
#                     letter_buffer.clear()
#                     hand_present = False
#
#                 cv2.rectangle(annotated_frame,(20,20),(SCREEN_SIZE[0]-20,150),(0,0,0),-1)
#                 cv2.putText(annotated_frame,f"Word:{recognized_text}",(30,80),
#                             cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,255,0),3)
#
#                 cv2.imshow("ASL Spell Out",annotated_frame)
#
#                 key = cv2.waitKey(1) & 0xFF
#                 if key == ord("q"):
#                     timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#                     snapshot_path = f"snapshots/{recognized_text}_{timestamp}.jpg"
#                     cv2.imwrite(snapshot_path, annotated_frame)
#                     print(f"Snapshot saved at{snapshot_path}")
#                     break
#                 elif key == ord('d'):
#                     if recognized_text:
#                         recognized_text =recognized_text[:-1]
#                         print("Deleted last letter")
#
#
#     cap.release()
#     cv2.destroyAllWindows()




from ultralytics import YOLO
import cv2
import os
import datetime

SCREEN_SIZE = (1280, 720)

model = YOLO(r"D:\PythonProject\ProjectASL\hand.pt")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_SIZE[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_SIZE[1])

recognized_text = ""
current_letter = None
stable_letter = None
hand_present = False

letter_buffer = []
BUFFER_SIZE = 10  # Number of frames needed for stability

os.makedirs("snapshots", exist_ok=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, conf=0.5, verbose=False)
    annotated_frame = results[0].plot()

    if results[0].boxes:
        class_id = int(results[0].boxes.cls[0])
        letter_buffer.append(model.names[class_id])

        if len(letter_buffer) > BUFFER_SIZE:
            letter_buffer.pop(0)

        # If the buffer mostly agrees → stable
        if letter_buffer.count(letter_buffer[-1]) > BUFFER_SIZE // 2:
            stable_letter = letter_buffer[-1]
            hand_present = True
    else:
        if hand_present and stable_letter:
            recognized_text += stable_letter
            print(f"✅ Finalized letter: {stable_letter}")
            stable_letter = None
            letter_buffer.clear()
        hand_present = False

    # Draw UI
    cv2.rectangle(annotated_frame, (20, 20), (SCREEN_SIZE[0] - 20, 150), (0, 0, 0), -1)
    cv2.putText(annotated_frame, f"Word: {recognized_text}", (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

    if stable_letter:
        cv2.putText(annotated_frame, f"Preview: {stable_letter}", (30, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)

    cv2.imshow("ASL Spell Out", annotated_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_path = f"snapshots/{recognized_text}_{timestamp}.jpg"
        cv2.imwrite(snapshot_path, annotated_frame)
        print(f"✅ Snapshot saved at {snapshot_path}")
        break
    elif key == ord('d'):
        if recognized_text:
            recognized_text = recognized_text[:-1]
            print("🗑 Deleted last letter")

cap.release()
cv2.destroyAllWindows()