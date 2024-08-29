import os
import dlib
import cv2
import numpy as np
import scipy

PREDICTOR_PATH = './shape_predictor_68_face_landmarks.dat'
#root_dir = '../guoetal-arraydata-tnf/Data/'
root_dir = '../celeba'
predictor = dlib.shape_predictor(PREDICTOR_PATH)

subdirs = os.listdir(root_dir)
'''temp = []
for subdir in subdirs:
    if subdir.endswith('graph'):
        temp.append(subdir)

subdirs = temp'''

temp = []
for subdir in subdirs:
    if subdir.endswith('crop'):
        temp.append(subdir)

subdirs = temp

print("Checking:", subdirs)

if not os.path.exists(root_dir + os.sep + 'stroke_landmarks'):
    os.makedirs(root_dir + os.sep + 'stroke_landmarks')
if not os.path.exists(root_dir + os.sep + 'nonstroke_landmarks'):
    os.makedirs(root_dir + os.sep + 'nonstroke_landmarks')

for subdir in subdirs:
    if subdir.startswith('stroke'):
        for item in os.listdir(root_dir + os.sep + subdir):
            image = cv2.imread(root_dir + os.sep + subdir + os.sep + item)
            landmarks = np.array(np.matrix([[p.x, p.y] for p in predictor(image, dlib.rectangle(0, 0, image.shape[0], image.shape[1])).parts()]))
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

            np.savez(root_dir + os.sep + f'stroke_landmarks/{item}.npz', l=landmarks, e=edge_index)
    if subdir.startswith('nonstroke'):
        for item in os.listdir(root_dir + os.sep + subdir):
            image = cv2.imread(root_dir + os.sep + subdir + os.sep + item)
            landmarks = np.array(np.matrix([[p.x, p.y] for p in predictor(image, dlib.rectangle(0, 0, image.shape[0], image.shape[1])).parts()]))
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

            np.savez(root_dir + os.sep + f'nonstroke_landmarks/{item}.npz', l=landmarks, e=edge_index)

    