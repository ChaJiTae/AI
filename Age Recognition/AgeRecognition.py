import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# 얼굴 인식용 Haar Cascade 파일 로드
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 나이 추정 모델 로드 (사전 훈련된 모델 필요)
age_model = load_model('age_model.h5')  # 모델 파일의 경로를 지정하세요

# 나이대 범위
age_ranges = ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60-100']

# 카메라 캡처 시작
cap = cv2.VideoCapture(0)

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # 그레이스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 얼굴 인식
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        # 얼굴 영역 추출 및 전처리
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (64, 64))
        face = face.astype('float') / 255.0
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)
        
        # 나이대 예측
        age_pred = age_model.predict(face)[0]
        age_range = age_ranges[np.argmax(age_pred)]
        
        # 얼굴 영역에 테두리 및 나이대 표시
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, age_range, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # 결과 프레임 출력
    cv2.imshow('Age Estimation', frame)
    
    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
