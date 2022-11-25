import cv2
import mediapipe as mp
import time
cap = cv2.VideoCapture(0)
pTime = 0
NUM_FACE = 2

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=NUM_FACE, refine_landmarks=True)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

id = 0
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms,mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)
            for id,lm in enumerate(faceLms.landmark):
                lm = faceLms.landmark[id]
                # print(lm)
                ih, iw, ic = img.shape
                x,y = int(lm.x*iw), int(lm.y*ih)
                # uncomment the below line to see the 468 facial landmark
                cv2.putText(img, str(id), (x,y), cv2.FONT_HERSHEY_PLAIN, 0.5, (0,255,0), 1)
                cv2.circle(img, (x,y), 2, (255,0,0), cv2.FILLED)
                # print(id, x,y)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS:{int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Test", cv2.resize(img, (2*640, 2*480)))
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('n'):
        id += 1
        if id > 477:
            id = 0
    if key == ord('p'):
        id -= 1
        if id < 0:
            id = 477