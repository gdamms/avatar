import cv2
import numpy as np
import mediapipe as mp


NOSE = 4
LEFT = 93
RIGHT = 323


class PointsDetector:
    """Landmarks detecttion using mediapipe face mesh."""

    def __init__(self) -> None:
        """Initialize mediapipe face mesh detector."""

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True)

        self.prev_points = []

    def raw_process(self, img: np.ndarray) -> np.ndarray:
        """Process image and return points.

        Args:
            img (np.ndarray): Image to process.

        Returns:
            np.ndarray: Nx2 matrix of points.
        """

        # Process image with mediapipe face mesh
        result = self.face_mesh.process(img)
        if not result.multi_face_landmarks:
            return None

        # Convert points to matrix
        mesh_points = self.convert(
            result.multi_face_landmarks[0].landmark, img.shape)

        return mesh_points

    def process(self, img: np.ndarray) -> np.ndarray:
        """Process image and return points.

        Args:
            img (np.ndarray): Image to process.

        Returns:
            np.ndarray: Nx2 matrix of points.
        """

        # Get raw points
        mesh_points = self.raw_process(img)
        if mesh_points is None:
            return None

        # Compute origin and normal of the face plane
        origin_point = (mesh_points[LEFT] + mesh_points[RIGHT]) / 2
        normal = mesh_points[NOSE] - origin_point
        normal = normal / np.linalg.norm(normal)

        # Project points on the face plane
        mesh_points = mesh_points - origin_point
        mesh_points = mesh_points - \
            np.dot(mesh_points, normal)[:, np.newaxis] * normal

        # Project points on the frame plane
        mesh_points = mesh_points + origin_point

        # Smooth mesh points
        self.prev_points.append(mesh_points[:, :2])
        if len(self.prev_points) > 5:
            self.prev_points.pop(0)
        mesh_points = np.mean(self.prev_points, axis=0)

        return mesh_points

    def convert(self, points: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        """Convert mediapipe points to matrix.

        Args:
            points (np.ndarray): mediapipe points
            shape (tuple): image shape

        Returns:
            np.ndarray: matrix of points
        """

        return np.array([[point.x*shape[1], point.y*shape[0], point.z*shape[1]] for point in points])


def rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image.

    Args:
        img (np.ndarray): image
        angle (float): angle (in radians)

    Returns:
        np.ndarray: rotated image
    """

    height, width = img.shape[:2]
    image_center = (width / 2, height / 2)

    # Get rotation matrix
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle*180/np.pi, 1.0)

    # Rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # Find the new width and height
    new_width = int(height * abs_sin + width * abs_cos)
    new_height = int(height * abs_cos + width * abs_sin)

    # Move image to fit new dimensions
    rotation_mat[0, 2] += new_width / 2 - image_center[0]
    rotation_mat[1, 2] += new_height / 2 - image_center[1]

    # Rotate the image
    rotated_mat = cv2.warpAffine(img, rotation_mat, (new_width, new_height))

    return rotated_mat


def overlay_image_alpha(img_source: np.ndarray, img_overlay: np.ndarray, x: int | float, y: int | float) -> np.ndarray:
    """Overlay img_overlay on top of img_source at the position (x, y), using alpha
    channel of img_overlay.

    Args:
        img_source (np.ndarray): background image
        img_overlay (np.ndarray): overlay image
        x (int | float): x position
        y (int | float): y position

    Returns:
        np.ndarray: overlayed image
    """

    # Setup
    x, y = int(x), int(y)
    img = img_source.copy()

    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return img

    # Blend overlay within the determined ranges
    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
    alpha = img_overlay[y1o:y2o, x1o:x2o, 3][:, :, np.newaxis] / 255.0
    alpha_inv = 1.0 - alpha

    img_crop[:] = alpha * img_overlay_crop + alpha_inv * img_crop

    return img
