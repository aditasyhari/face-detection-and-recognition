import imutils
import cv2
import os

face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade.xml')

def face_extractor(origin, destination, fc):
    ## Importing image using open cv
    img = cv2.imread(origin,1)

    ## Resizing to constant width
    img = imutils.resize(img, width=200)
    
    ## Finding actual size of image
    H,W,_ = img.shape
    
    ## Converting BGR to RGB
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ## Detecting faces on the image
    face_coord = fc.detectMultiScale(gray,1.2,10,minSize=(50,50))
    
    ## If only one face is foung
    if len(face_coord) == 1:
        X, Y, w, h = face_coord[0]
    
    ## If no face found --> SKIP
    elif len(face_coord)==0:
        return None
    
    ## If multiple faces are found take the one with largest area
    else:
        max_val = 0
        max_idx = 0
        for idx in range(len(face_coord)):
            _, _, w_i, h_i = face_coord[idx]
            if w_i*h_i > max_val:
                max_idx = idx
                max_val = w_i*h_i
            else:
                pass
            
            X, Y, w, h = face_coord[max_idx]
    
    ## Crop and export the image
    # img_cp = img[
    #         max(0,Y - int(0.35*h)): min(Y + int(1.35*h), H),
    #         max(0,X - int(w*0.35)): min(X + int(1.35*w), W)
    #     ].copy()

    img_cp = img[
        Y:Y+h, X:X+w
    ].copy()
    
    cv2.imwrite(destination, img_cp)


## Defining destination path
directory = 'dataset/face/'
dir = os.listdir(directory)
labels = [""]

for item in dir:
    if os.path.isdir(directory+item) == True:
        labels.append(item)

print(labels)
# print(labels[4])

# labels = sorted(labels)
# print(labels)

for label in labels:
    source_dir = 'dataset/cropped/'+label+'/'
    if os.path.exists(source_dir) == False:
        os.makedirs(source_dir)

no = 0
for path, subdirs, files in os.walk(directory):
    print(path)
    for file in files:
        face_extractor(origin = path+"/"+file, destination = 'dataset/cropped/'+labels[no]+'/'+file, fc=face_cascade)
        # print(subdirs)
        # print(file)

    no+=1