import os
import cv2
import numpy as np
import mediapipe as mp


MOUTH_IN = [13, 14]
MOUTH_OUT = [0, 17, 37, 39]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154,
             155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390,
            249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_BROW = []
LEFT_BROW = []
NOSE = [1]
FACE = [10, 21]
RIGHT_IRIS = [468, 469, 470, 471, 472]
LEFT_IRIS = [473, 474, 475, 476, 477]


class PointsDetector:
    """Landmarks detecttion using mediapipe face mesh."""

    def __init__(self) -> None:
        """Initialize mediapipe face mesh detector."""

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True)

    def process(self, img: np.ndarray) -> dict:
        """Process image and return points.

        Args:
            img (np.ndarray): Image to process.

        Returns:
            dict: Dictionary with points.
        """

        # Process image with mediapipe face mesh
        result = self.face_mesh.process(img)
        if not result.multi_face_landmarks:
            return None

        # Convert points to matrix
        mesh_points = self.convert(
            result.multi_face_landmarks[0].landmark, img.shape)

        return {
            'mouth_in': mesh_points[MOUTH_IN],
            'mouth_out': mesh_points[MOUTH_OUT],
            'right_eye': mesh_points[RIGHT_EYE],
            'left_eye': mesh_points[LEFT_EYE],
            'eyeR': mesh_points[RIGHT_IRIS],
            'eyeL': mesh_points[LEFT_IRIS],
            'nose': mesh_points[NOSE],
            'face': mesh_points[FACE]
        }

    def convert(self, points: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        """Convert mediapipe points to matrix.

        Args:
            points (np.ndarray): mediapipe points
            shape (tuple): image shape

        Returns:
            np.ndarray: matrix of points
        """

        return np.array([[int(point.x*shape[1]), int(point.y*shape[0])] for point in points])


def compute_transformation(ref_points: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute transformation parameters.

    Args:
        ref_points (np.ndarray): reference points
        points (np.ndarray): points

    Returns:
        tuple[np.ndarray, np.ndarray, float]: transformation parameters
    """

    X = np.array([
        ref_points[:, 0],
        ref_points[:, 1],
        np.ones(len(ref_points))
    ])
    b = np.array([
        points[:, 0],
        points[:, 1]
    ])

    # Solve linear system
    A = b @ np.linalg.pinv(X)

    # Extract transformation parameters
    tx = A[0, 2]
    ty = A[1, 2]

    # Compute squeeze factors
    Sx = np.sqrt(A[0, 0]**2 + A[1, 0]**2)
    Sy = np.sqrt(A[0, 1]**2 + A[1, 1]**2)

    # Compute rotation angle
    angle = -np.arctan2(A[1, 0], A[0, 0])

    return np.array([tx, ty]), np.array([Sx, Sy]), angle


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


def apply_transformation(img: np.ndarray, translation: np.ndarray, squeeze: np.ndarray, angle: float) -> np.ndarray:
    # Apply squeeze
    img = cv2.resize(img, (0, 0), fx=squeeze[0], fy=squeeze[1])

    # Apply rotation
    img = rotate_image(img, angle)

    # Correct translation from rotation
    img_center = np.array([img.shape[1]//2, img.shape[0]//2])
    translation -= img_center

    # Apply translation
    img = cv2.copyMakeBorder(img, int(translation[1]), 0, int(translation[0]), 0,
                             cv2.BORDER_CONSTANT, value=(0, 0, 0))

    return img


def overlay_image_alpha(img_source: np.ndarray, img_overlay: np.ndarray, x: float, y: float) -> np.ndarray:
    """Overlay img_overlay on top of img_source at the position (x, y), using alpha
    channel of img_overlay.

    Args:
        img_source (np.ndarray): background image
        img_overlay (np.ndarray): overlay image
        x (float): x position
        y (float): y position

    Returns:
        np.ndarray: overlayed image
    """

    img = img_source.copy()

    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    # Blend overlay within the determined ranges
    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
    alpha = img_overlay[y1o:y2o, x1o:x2o, 3][:, :, np.newaxis] / 255.0
    alpha_inv = 1.0 - alpha

    img_crop[:] = alpha * img_overlay_crop + alpha_inv * img_crop

    return img


def _main_():
    
    # Reference points for each piece of the face
    image_points = {
        'eyeL': np.array([[0.0, 0.0], [17.5, 0.0], [0.0, -17.5], [-17.5, 0.0], [0.0, 17.5]]),
        'eyeR': np.array([[0.0, 0.0], [17.5, 0.0], [0.0, -17.5], [-17.5, 0.0], [0.0, 17.5]]),
    }

    # Initialize tools
    point_detector = PointsDetector()
    cap = cv2.VideoCapture(0)

    running = True
    while running:
        # Capture and process frame
        success, img_source = cap.read()
        img_source = cv2.cvtColor(img_source, cv2.COLOR_BGR2BGRA)
        points = point_detector.process(img_source[..., :3])

        # Skip frame if no face detected
        if points is None:
            continue

        # Draw the avatar
        img_avatar = 255 * np.ones_like(img_source)
        for img_name in image_points:
            img_path = os.path.join('imgs', 'avatar', img_name + '.png')
            avatar_piece = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            translation, squeeze, angle = compute_transformation(
                image_points[img_name], points[img_name])
            avatar_piece = apply_transformation(
                avatar_piece, translation, squeeze, angle)
            img_avatar = overlay_image_alpha(img_avatar, avatar_piece, 0, 0)

        # Display the result
        cv2.imshow('Source', img_source)
        cv2.imshow('Avatar', img_avatar)

        # Exit on `q` or `esc`
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:
            running = False
    cap.release()


if __name__ == '__main__':
    _main_()
