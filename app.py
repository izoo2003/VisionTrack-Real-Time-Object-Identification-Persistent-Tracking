import cv2
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

def run_tracker():
    # Load model and tracker
    model = YOLO('yolov8n.pt')
    tracker = DeepSort(max_age=30, n_init=3)
    
    # model.names is a dict: {0: 'person', 1: 'bicycle', 67: 'cell phone', ...}
    class_names = model.names 

    cap = cv2.VideoCapture(0)
    prev_time = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        # 1. Detection
        results = model(frame)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            if conf > 0.6:
                detections.append([[x1, y1, x2-x1, y2-y1], conf, cls])

        # 2. Tracking
        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed(): continue
            
            track_id = track.track_id
            # Get the original class ID stored in the track
            class_id = track.get_det_class()
            # Look up the name (e.g., 'cell phone') from our dictionary
            object_name = class_names[class_id]
            
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            
            # Use a unique color for each ID 
            color = (0, 255, 0) 
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Create label: "cell phone #ID"
            display_text = f"{object_name.upper()} #{track_id}"
            
            cv2.putText(frame, display_text, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 3. FPS Display
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow('Object Identification & Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_tracker()