def analyze_video_file(self, filepath):
    cap = cv2.VideoCapture(filepath)
    scores = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # 1. Detect and Crop the face
        face_crop = self.extract_face(frame) 
        
        if face_crop is not None:
            # 2. Pass the crop to your actual MesoNet/Deepfake model
            prediction = self.model.predict(face_crop) 
            scores.append(prediction)
            
    cap.release()
    return sum(scores) / len(scores) if scores else 0.0
