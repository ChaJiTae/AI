import cv2
import numpy as np
import tensorflow as tf

# 사전 학습된 모델 로드
age_model = tf.keras.models.load_model('age_model.h5')
gender_model = tf.keras.models.load_model('gender_model.h5')

# Haar Cascade 파일 로드
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 얼굴 이미지 전처리
def preprocess_face(face):
    face = cv2.resize(face, (64, 64))
    face = face.astype("float") / 255.0
    face = np.expand_dims(face, axis=0)
    return face

# 나이 및 성별 예측
def predict_age_gender(face_image):
    processed_face = preprocess_face(face_image)
    age = age_model.predict(processed_face)
    gender = gender_model.predict(processed_face)
    gender_label = 'Male' if gender[0][0] > 0.5 else 'Female'
    return int(age[0][0]), gender_label

# 비디오 캡처 초기화
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    predictions = []
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        age, gender = predict_age_gender(face)
        predictions.append(((x, y, w, h), (age, gender)))
        
        # 얼굴 및 예측 결과 그리기
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f'{gender}, {age}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    cv2.imshow('Age and Gender Prediction', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
