from enum import EnumMeta
import numpy as np
import dlib
import cv2

RIGHT_EYE=list(range(36,42))
LEFT_EYE=list(range(42,48))
MOUTH=list(range(48,68))
NOSE=list(range(27,36))
EYEBROWS=list(range(17,27))
JAWLINE=list(range(17,27))
ALL=list(range(0,68))

# 학습된 모델로 dlib에서 미리 만들어놓은 데이터이다.
predictor_file='./model/shape_predictor_68_face_landmarks.dat'
image_file='./image/1people.jpg'

# 정면 사진을 감지
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor(predictor_file)

image=cv2.imread(image_file)
#image=cv2.resize(image,(1080,720))
# 흑백으로 바꾸어서 단순화 시켜 인식률을 높임
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# 1 -> detection 하기 전에 layer를 upscale하는데 몇번 적용할지
rects=detector(gray,1)
print('Number of faces detected: {}'.format(len(rects)))




for (i,rect) in enumerate(rects):
    points=np.matrix([[p.x,p.y] for p in predictor(gray,rect).parts()]) # 68개 점의 좌표
    show_parts=points[ALL] # points[x]로 내가 원하는 부분만 검출 가능    
    
    for (i,point) in enumerate(show_parts):
        x=point[0,0]
        y=point[0,1]
        cv2.circle(image,(x,y),1,(0,255,255),-1)
        cv2.putText(image,'{}'.format(i+1),(x,y-2),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,0.3,(0,255,0),1)

image=cv2.resize(image,(1080,720))
cv2.imshow('Face landmark',image)
cv2.waitKey(0)

