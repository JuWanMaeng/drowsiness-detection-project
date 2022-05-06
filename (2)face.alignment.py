import numpy as np
import dlib
import cv2

RIGHT_EYE=list(range(36,42))
LEFT_EYE=list(range(42,48))
EYES=list(range(36,48))

predictor_file = './model/shape_predictor_68_face_landmarks.dat'
image_file='./image/tilt.jpg'
MARGIN_RATIO=1.5  # 실제 얼굴 크기보다 추출된 사진의 크기 비율 계수
OUTPUT_SIZE=(300,300)

detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor(predictor_file)

image=cv2.imread(image_file)
image_origin=image.copy()

(image_height,image_width) = image.shape[:2]
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

rects=detector(gray,1)

def getFaceDimension(rect):
    # 얼굴을 가리키는 사각형의 왼쪽 상단 (꼭지점 좌표, 너비, 높이)
    return (rect.left(),rect.top(),rect.right()-rect.left(),rect.bottom()-rect.top())
def getCropDimension(rect,center):
    width=(rect.right() - rect.left())
    half_width=width//2
    (centerX,centerY) = center
    startX=centerX-half_width
    endX=centerX + half_width
    startY=rect.top()
    endY=rect.bottom()
    return (startX,endX,startY,endY)

for (i,rect) in enumerate(rects):
    (x,y,w,h) = getFaceDimension(rect)
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    
    points=np.matrix([[p.x,p.y] for p in predictor(gray,rect).parts()])
    show_parts=points[EYES]
    
    right_eye_center=np.mean(points[RIGHT_EYE],axis=0).astype('int')  # 오른쪽 눈의 좌표값들의 평균값
    left_eye_center=np.mean(points[LEFT_EYE],axis=0).astype('int') # 왼쪽 눈의 좌표값들의 평균값
    print(right_eye_center,left_eye_center)
    
    cv2.circle(image,(right_eye_center[0,0],right_eye_center[0,1]),5,(0,0,255),-1)
    cv2.circle(image,(left_eye_center[0,0],left_eye_center[0,1]),5,(0,0,255),-1)
    
    # 왼쪽눈의 x좌표, 오른쪽 눈의 y좌표(실제 방향)
    cv2.circle(image,(left_eye_center[0,0],right_eye_center[0,1]),5,(0,255,0),-1)
    
    cv2.line(image,(right_eye_center[0,0],right_eye_center[0,1]),
    (left_eye_center[0,0],left_eye_center[0,1]),(0,255,0),2)
    cv2.line(image,(right_eye_center[0,0],right_eye_center[0,1]),
    (left_eye_center[0,0],right_eye_center[0,1]),(0,255,0),1)
    cv2.line(image,(left_eye_center[0,0],right_eye_center[0,1]),
    (left_eye_center[0,0],left_eye_center[0,1]),(0,255,0),1)
    
    eye_delta_x=right_eye_center[0,0] - left_eye_center[0,0]
    eye_delta_y=right_eye_center[0,1] - left_eye_center[0,1]
    degree=np.degrees(np.arctan2(eye_delta_y,eye_delta_x)) - 180
    #print(degree)
    
    eye_distance=np.sqrt((eye_delta_x ** 2) + (eye_delta_y ** 2))
    # 사진을 돌린 이후 눈 사이의 거리 조정
    aligned_eye_distance=left_eye_center[0,0] - right_eye_center[0,0]
    scale=aligned_eye_distance/eye_distance

    eyes_center=((left_eye_center[0,0] + right_eye_center[0,0])//2,
    (left_eye_center[0,1] + right_eye_center[0,1])//2)

    cv2.circle(image,eyes_center,5,(255,0,0),-1)
    
    # 영상의 중앙 기준 회전
    matrix=cv2.getRotationMatrix2D(eyes_center,degree,scale)
    cv2.putText(image,'{:.5f}'.format(degree),(right_eye_center[0,0],right_eye_center[0,1]),
    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)  
    
    wraped=cv2.warpAffine(image_origin,matrix,(image_width,image_height),
                          flags=cv2.INTER_CUBIC)
    
    cv2.imshow('wrapAffine',wraped)
    cv2.waitKey(0)
    (startX,endX,startY,endY) =getCropDimension(rect,eyes_center)
    cropped=wraped[startY:endY,startX:endX]
    output=cv2.resize(cropped,OUTPUT_SIZE)
    cv2.imshow('output',output)
    cv2.waitKey(0)
  
cv2.imshow('Face Alignment',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
    
    