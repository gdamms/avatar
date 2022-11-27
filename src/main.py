import os
import cv2
import json
import argparse
from rich_argparse import RichHelpFormatter
import numpy as np

from utils import *
from calibration import calibrate_avatar


def compute_transformation(
    ref_points: np.ndarray,
    points: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
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


def apply_transformation(
    img_source: np.ndarray,
    img: np.ndarray,
    translation: np.ndarray,
    squeeze: np.ndarray,
    angle: float
) -> np.ndarray:
    """Apply transformation of img to img_source.

    Args:
        img_source (np.ndarray): source image
        img (np.ndarray): image to transform
        translation (np.ndarray): translation vector
        squeeze (np.ndarray): squeeze factors
        angle (float): rotation angle

    Returns:
        np.ndarray: transformed image
    """

    # Apply squeeze
    img = cv2.resize(img, (0, 0), fx=squeeze[0], fy=max(squeeze[1], 0.01))

    # Apply rotation
    img = rotate_image(img, angle)

    # Correct translation from rotation
    img_center = np.array([img.shape[1] / 2, img.shape[0] / 2])
    translation -= img_center

    # Apply translation and overlay
    img_result = overlay_image_alpha(
        img_source, img, int(translation[0]), int(translation[1]))

    return img_result


def _main_(args: argparse.Namespace) -> None:
    """Main function."""

    path = os.path.join(os.path.dirname(__file__), 'data', args.avatar)
    
    # Calibration
    if args.calibrate:
        calibrated = calibrate_avatar(path)
        if not calibrated:
            return

    # Load json config file
    with open(os.path.join(path, 'config.json')) as f:
        avatar_config = json.load(f)

    # Initialize tools
    point_detector = PointsDetector()
    cap = cv2.VideoCapture(0)

    # Create windows
    cv2.namedWindow('Source', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Avatar', cv2.WINDOW_NORMAL)

    running = True
    # Main loop
    while running:
        # Capture and process frame
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        points = point_detector.process(frame[..., :3])

        # Skip frame if no face detected
        if points is None:
            continue

        # Draw the avatar
        img_avatar = 255 * np.ones_like(frame)
        for name, piece in avatar_config['pieces'].items():
            # Get current piece of the face
            img_path = os.path.join(path, name + '.png')
            avatar_piece = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

            # Get transformation parameters
            translation, squeeze, angle = compute_transformation(
                np.array(piece['calibration']),
                points[piece['mesh']])

            # Apply transformation
            img_avatar = apply_transformation(
                img_avatar,
                avatar_piece,
                translation,
                squeeze,
                angle)

        # Display the result
        cv2.imshow('Source', frame)
        cv2.imshow('Avatar', img_avatar)

        # Handle keyboard events
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:  # q or ESC
            running = False

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(formatter_class=RichHelpFormatter)
    parser.add_argument('-a', '--avatar',
                        type=str, default='avatar', help='avatar name')
    parser.add_argument('-c', '--calibrate',
                        action='store_true', help='calibrate avatar')
    args = parser.parse_args()

    # Run main function
    _main_(args)
