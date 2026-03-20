import cv2
import numpy as np
import os

def main():
    print("Loading models...")
    
    # Check if models exist
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prototxt_path = os.path.join(script_dir, 'models', 'age_deploy.prototxt')
    model_path = os.path.join(script_dir, 'models', 'age_net.caffemodel')
    
    if not os.path.exists(prototxt_path) or not os.path.exists(model_path):
        print(f"Error: Could not find model files at {os.path.join(script_dir, 'models')}. Please run 'python download_models.py' first.")
        return

    # Load age network
    try:
        age_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    except Exception as e:
        print(f"Error loading models. They might be corrupted or incomplete: {e}")
        return
        
    # Age ranges as defined by the original authors
    AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    
    # Load face cascade
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    
    print("Initializing webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open the webcam. Make sure it's connected and not used by another app.")
        return

    print("Webcam started successfully. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Optional: mirror the frame a natural feel
        frame = cv2.flip(frame, 1)    
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            # Draw bounding box for face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Extract face ROI with a slightly larger box for better age estimation context
            padding = 20
            roi_y1 = max(0, y - padding)
            roi_y2 = min(frame.shape[0] - 1, y + h + padding)
            roi_x1 = max(0, x - padding)
            roi_x2 = min(frame.shape[1] - 1, x + w + padding)
            
            face_img = frame[roi_y1:roi_y2, roi_x1:roi_x2].copy()
            if face_img.shape[0] < 10 or face_img.shape[1] < 10:
                continue
                
            # Prepare image for the DNN
            # The model expects 227x227 shape, and specific mean subtraction
            blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
            
            # Predict age
            age_net.setInput(blob)
            preds = age_net.forward()
            i = preds[0].argmax()
            age = AGE_LIST[i]
            age_confidence = preds[0][i]
            
            # Display age with confidence
            label = f"Age: {age} ({age_confidence*100:.1f}%)"
            
            # Nice background for text readability
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x, y - label_size[1] - 10), (x + label_size[0], y), (255, 0, 0), cv2.FILLED)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
        cv2.imshow('Age Guesser (Press Q to quit)', frame)
        
        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")

if __name__ == "__main__":
    main()
