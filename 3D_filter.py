import cv2
import time
import mediapipe as mp
from math import hypot, atan2, degrees
import numpy as np


# Simple OBJ loader class with texture support
class OBJ:
    def __init__(self, filename, swapyz=False):
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []
        for line in open(filename, "r"):
            if line.startswith('#'):
                continue
            values = line.split()
            if not values:
                continue
            if values[0] == 'v':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.vertices.append(v)
            elif values[0] == 'vn':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.normals.append(v)
            elif values[0] == 'vt':
                self.texcoords.append(list(map(float, values[1:3])))
            elif values[0] == 'f':
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)
                self.faces.append((face, norms, texcoords))


# Load the OBJ file
obj_model = OBJ('mustache.obj')

# Load the texture image
texture_image = cv2.imread('texture_rgb.png')
texture_h, texture_w, _ = texture_image.shape

# Calculate the original width and height of the OBJ model
original_width = max([v[0] for v in obj_model.vertices]) - min([v[0] for v in obj_model.vertices])
original_height = max([v[1] for v in obj_model.vertices]) - min([v[1] for v in obj_model.vertices])

# Function to render the OBJ model with dynamic scaling and texture
def render_obj_dynamic_scale(obj, image, position, scale_x=1.0, scale_y=1.0, angle=0):
    overlay = image.copy()
    for face in obj.faces:
        vertices = [obj.vertices[idx - 1] for idx in face[0]]
        vertices = np.array(vertices)
        texcoords = [obj.texcoords[idx - 1] for idx in face[2]]

        # Scale the vertices
        vertices[:, 0] *= scale_y
        vertices[:, 1] *= scale_x

        # Rotate the model according to the calculated angle
        cos_angle = np.cos(np.radians(angle))
        sin_angle = np.sin(np.radians(angle))
        rotation_matrix = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])
        vertices[:, :2] = np.dot(vertices[:, :2], rotation_matrix)

        # Adjust the position to the calculated point
        pts = vertices[:, :2].astype(int) + np.array(position)

        # Map the texture coordinates to the texture image
        texcoords = np.array(texcoords)
        texcoords[:, 0] = (texcoords[:, 0] * (texture_w - 1)).astype(int)
        texcoords[:, 1] = (1 - texcoords[:, 1]) * (texture_h - 1)  # Flip the Y-axis of texture coordinates
        texcoords = texcoords.astype(int)

        # Get the texture colors for the face vertices
        color = np.mean([texture_image[texcoord[1], texcoord[0]] for texcoord in texcoords], axis=0)

        # Draw the textured polygon on the overlay
        cv2.fillPoly(overlay, [pts], color)

    # Blend the overlay with the original image
    cv2.addWeighted(overlay, 1.0, image, 0.0, 0, image)


# Use the default camera as the video source
cap = cv2.VideoCapture(0)

# Initialize Mediapipe FaceMesh
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)

previous_center_lip = None  # Store the previous center position

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mediapipe needs RGB image
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = faceMesh.process(rgb)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, c = rgb.shape

            # Extract the current center lip position
            center_lip = (int(face_landmarks.landmark[164].x * w), int(face_landmarks.landmark[164].y * h))

            # Adjust the vertical position to bring the mustache lower
            center_lip = (center_lip[0], center_lip[1] + 10)  # Move the mustache 10 pixels downward

            # If there is a previous position, calculate the movement
            if previous_center_lip is not None:
                movement_x = center_lip[0] - previous_center_lip[0]
                movement_y = center_lip[1] - previous_center_lip[1]
                # Apply movement to the current center position
                center_lip = (previous_center_lip[0] + movement_x, previous_center_lip[1] + movement_y)

            # Update the previous center lip position
            previous_center_lip = center_lip

            # Calculate the width and height of the mustache based on the distance between the lip corners
            left_lip = (int(face_landmarks.landmark[287].x * w), int(face_landmarks.landmark[287].y * h))
            right_lip = (int(face_landmarks.landmark[57].x * w), int(face_landmarks.landmark[57].y * h))
            mustache_width = int(hypot(left_lip[0] - right_lip[0], left_lip[1] - right_lip[1]) * 1.2)
            mustache_height = int(mustache_width * 0.3)  # Reduced height (from 0.5 to 0.3)

            # Calculate the angle between the lip corners (horizontal angle for side detection)
            angle = atan2(right_lip[1] - left_lip[1], right_lip[0] - left_lip[0])
            angle_deg = -(degrees(angle))

            # Only apply scaling if the horizontal angle indicates a significant side profile
            scale_x = mustache_width / original_width
            scale_y = mustache_height / original_height
            render_obj_dynamic_scale(obj_model, frame, center_lip, scale_x, scale_y, angle_deg)

    # Display the output frame
    cv2.imshow("output", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
