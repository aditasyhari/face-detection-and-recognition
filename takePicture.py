import cv2 

key = cv2.waitKey(1)
webcam = cv2.VideoCapture(0)
number = 1
while True:
    try:
        check, frame = webcam.read()
        frame = cv2.flip(frame, 1)
        print(check)
        print(frame)
        cv2.imshow("Capturing", frame)
        key = cv2.waitKey(1)
        if key == ord('s'): 
            cv2.imwrite(filename='dataset/face/Aditya R/'+str(number)+'_aditya_img.jpg', img=frame)
            number+=1
            
        elif key == ord('q'):
            print("Mematikan kamera.")
            webcam.release()
            print("Kamera mati.")
            print("Program selesai.")
            cv2.destroyAllWindows()
            break
        
    except(KeyboardInterrupt):
        print("Mematikan kamera.")
        webcam.release()
        print("Kamera mati.")
        print("Program selesai.")
        cv2.destroyAllWindows()
        break