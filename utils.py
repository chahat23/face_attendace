import numpy as np
from keras import config
config.enable_unsafe_deserialization()


def preprocess_face(face_img):
    """
    Normalize face image for FaceNet
    """
    face_img = face_img.astype('float32')
    mean, std = face_img.mean(), face_img.std()
    return (face_img - mean) / std


def cosine_distance(emb1, emb2):
    """
    Compute cosine distance between two embeddings
    """
    dot = np.dot(emb1, emb2)
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    return 1 - (dot / (norm1 * norm2))

