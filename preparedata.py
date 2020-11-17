## import libraries
import os, shutil
import numpy as np
import cv2
import matplotlib.pyplot as plt


from glob import glob
import warnings
warnings.filterwarnings('ignore')


## haar-cascade
path = '/home/kerrykim/jupyter_notebook/5. eyetracking/tmp/'

faceDet = cv2.CascadeClassifier(path + "haarcascade_frontalface_default.xml")
faceDet_two = cv2.CascadeClassifier(path + "haarcascade_frontalface_alt2.xml")
faceDet_three = cv2.CascadeClassifier(path + "haarcascade_frontalface_alt.xml")
faceDet_four = cv2.CascadeClassifier(path + "haarcascade_frontalface_alt_tree.xml")

eye_cascade = cv2.CascadeClassifier(path + 'haarcascade_eye.xml')


## extract face-features
def detect_eyes(face, img, gray):
    [x, y, w, h] = face
    roi_gray = gray[y:y + h, x:x + w]

    eyes = eye_cascade.detectMultiScale(roi_gray)
    eyes_sorted_by_size = sorted(eyes, key=lambda x: -x[2])
    largest_eyes = eyes_sorted_by_size[:2]

    # sort by x position
    largest_eyes.sort(key=lambda x: x[0])

    # offset by face start
    return list(map(lambda eye: [face[0] + eye[0], face[1] + eye[1], eye[2], eye[3]], largest_eyes))


def extract_image_features(index, image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5))
    face_two = faceDet_two.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5))
    face_three = faceDet_three.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5))
    face_four = faceDet_four.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5))

    if len(face) == 1:
        face_detections = face
    elif len(face_two) == 1:
        face_detections = face_two
    elif len(face_three) == 1:
        face_detections = face_three
    elif len(face_four) == 1:
        face_detections = face_four
    else:
        face_detections = ""

    # Make face check folder
    filename = os.path.join('./checkFace', 'input_%05d.jpg' % index)
    if face_detections == 1:
        shutil.copyfile(image_path, filename)

    left_to_right_face_detections = sorted(face_detections, key=lambda detection: detection[0])

    faces = []
    face_features = []

    for [x, y, w, h] in left_to_right_face_detections:
        face = [x, y, w, h]
        #  start_eyes = current_time()
        eyes = detect_eyes(face, img, gray)
        #  print('eye extraction '  + str((current_time() - start_eyes) / 1000.))
        face_grid = get_face_grid(face, img.shape[1], img.shape[0], 25)

        faces.append(face)
        face_features.append([eyes, face_grid])

    return img, faces, face_features


## define faceGrid
def get_face_grid(face, frameW, frameH, gridSize):
    faceX, faceY, faceW, faceH = face
    return faceGridFromFaceRect(frameW, frameH, gridSize, gridSize, faceX, faceY, faceW, faceH, False)

def faceGridFromFaceRect(frameW, frameH, gridW, gridH, labelFaceX, labelFaceY, labelFaceW, labelFaceH, parameterized):
    scaleX = gridW / frameW
    scaleY = gridH / frameH

    if parameterized:
        labelFaceGrid = np.zeros(4)
    else:
        labelFaceGrid = np.zeros(gridW * gridH)

    grid = np.zeros((gridH, gridW))

    # Use one-based image coordinates.
    xLo = round(labelFaceX * scaleX)
    yLo = round(labelFaceY * scaleY)
    w = round(labelFaceW * scaleX)
    h = round(labelFaceH * scaleY)

    if parameterized:
        labelFaceGrid = [xLo, yLo, w, h]
    else:
        xHi = xLo + w
        yHi = yLo + h

        # Clamp the values in the range.
        xLo = int(min(gridW, max(0, xLo)))
        xHi = int(min(gridW, max(0, xHi)))
        yLo = int(min(gridH, max(0, yLo)))
        yHi = int(min(gridH, max(0, yHi)))

        faceLocation = np.ones((yHi - yLo, xHi - xLo))
        grid[yLo:yHi, xLo:xHi] = faceLocation

        # Flatten the grid.
        grid = np.transpose(grid)
        labelFaceGrid = grid.flatten()

    return labelFaceGrid

## crop image
def crop_image(img, crop):
    return img[crop[1]:crop[1] + crop[3], crop[0]:crop[0] + crop[2], :]


def test_face(img, face, face_feature):
    eyes, faceGrid = face_feature

    if len(eyes) < 2:
        return None

    # crop
    imFace = crop_image(img, face)
    imEyeL = crop_image(img, eyes[1])
    imEyeR = crop_image(img, eyes[0])

    return imFace, imEyeL, imEyeR, faceGrid


def test_faces(path, img, faces, face_features):
    for i, face in enumerate(faces):
        imFace, imEyeL, imEyeR, faceGrid = test_face(img, face, face_features[i])

        plt.imsave(os.path.join(path+'appleFace', 'input_%05d.jpg' % i), imFace)
        plt.imsave(os.path.join(path+'appleLeftEye', 'input_%05d.jpg' % i), imEyeL)
        plt.imsave(os.path.join(path+'appleRightEye', 'input_%05d.jpg' % i), imEyeR)
        plt.imsave(os.path.join(path+'appleGrid', 'input_%05d.jpg' % i), faceGrid)


## main
if __name__ == "__main__":
    # Make folders
    path = '/home/kerrykim/jupyter_notebook/5. eyetracking/data_test/'

    folder_check = path + "checkFace"
    folder_face = path + "appleFace"
    folder_Leye = path + "appleLeftEye"
    folder_Reye = path + "appleRightEye"
    folder_Grid = path + "appleGrid"

    lst_folder = [folder_check, folder_face, folder_Leye, folder_Reye, folder_Grid]

    for fl in lst_folder:
        if not os.path.exists(fl):
            os.makedirs(fl)

    # get list of all test images
    lst_data = glob(path + 'origin')

    # extract face features
    for index, fl in enumerate(lst_data):
        img, faces, face_features = extract_image_features(index, fl)
        test_faces(path=path, img=img, faces=faces, face_features=face_features)

    print('DONE')