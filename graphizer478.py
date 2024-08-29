import os
import dlib
import cv2
import numpy as np
import scipy

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

#root_dir = '../guoetal-arraydata-tnf/Data/'
root_dir = '../celeba'

subdirs = os.listdir(root_dir)

temp = []
for subdir in subdirs:
    if subdir.endswith('crop'):
        temp.append(subdir)

subdirs = temp

print("Checking:", subdirs)

if not os.path.exists('stroke_landmarks'):
    os.makedirs('stroke_landmarks')
if not os.path.exists('nonstroke_landmarks'):
    os.makedirs('nonstroke_landmarks')

base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

for subdir in subdirs:
    if subdir.startswith('stroke'):
        for item in os.listdir(root_dir + os.sep + subdir):
            dim = cv2.imread(root_dir + os.sep + subdir + os.sep + item).shape
            image = mp.Image.create_from_file(root_dir + os.sep + subdir + os.sep + item)
            detection = detector.detect(image)
            landmarks = []
            if len(detection.face_landmarks) > 0:
                for landmark in detection.face_landmarks[0]:
                    landmarks.append((int(landmark.x * dim[1]), int(landmark.y * dim[0])))
                landmarks = np.asarray(landmarks)
                tri_edges = scipy.spatial.Delaunay(landmarks).simplices
                edges = []
                for simplex in tri_edges:
                    edges.append([simplex[0], simplex[1]])
                    edges.append([simplex[1], simplex[2]])
                    edges.append([simplex[2], simplex[0]])
                edges = np.asarray(edges)
                for i, edge in enumerate(edges):
                    temp = [min(edge[0], edge[1]), max(edge[0], edge[1])]
                    edges[i] = temp
                edge_index = np.unique(edges, axis=0)

                np.savez(f'stroke_landmarks/{item}.npz', l=landmarks, e=edge_index)
    if subdir.startswith('nonstroke'):
        for item in os.listdir(root_dir + os.sep + subdir):
            dim = cv2.imread(root_dir + os.sep + subdir + os.sep + item).shape
            image = mp.Image.create_from_file(root_dir + os.sep + subdir + os.sep + item)
            detection = detector.detect(image)
            landmarks = []
            if len(detection.face_landmarks) > 0:
                for landmark in detection.face_landmarks[0]:
                    landmarks.append((int(landmark.x * dim[1]), int(landmark.y * dim[0])))
                landmarks = np.asarray(landmarks)
                tri_edges = scipy.spatial.Delaunay(landmarks).simplices
                edges = []
                for simplex in tri_edges:
                    edges.append([simplex[0], simplex[1]])
                    edges.append([simplex[1], simplex[2]])
                    edges.append([simplex[2], simplex[0]])
                edges = np.asarray(edges)
                for i, edge in enumerate(edges):
                    temp = [min(edge[0], edge[1]), max(edge[0], edge[1])]
                    edges[i] = temp
                edge_index = np.unique(edges, axis=0)

                np.savez(f'nonstroke_landmarks/{item}.npz', l=landmarks, e=edge_index)

    