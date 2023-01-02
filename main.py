import os
import cv2
import json
import argparse
from rich_argparse import RichHelpFormatter
import numpy as np

from src.utils import *
from src.calibration import calibrate_avatar


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
    img = cv2.resize(img, (0, 0), fx=max(squeeze[0], 0.05), fy=max(squeeze[1], 0.05))

    # Apply rotation
    img = rotate_image(img, angle)

    # Correct translation from rotation
    img_center = np.array([img.shape[1] / 2, img.shape[0] / 2])
    translation -= img_center

    # Apply translation and overlay
    img_result = overlay_image_alpha(
        img_source, img, int(translation[0]), int(translation[1]))

    return img_result


def compute_homography(
    ref_points: np.ndarray,
    points: np.ndarray
) -> np.ndarray:
    """Compute homography parameters.

    Args:
        ref_points (np.ndarray): reference points
        points (np.ndarray): points

    Returns:
        np.ndarray: homography parameters
    """

    # Compute homographie
    H, _ = cv2.findHomography(ref_points, points, cv2.RANSAC)

    return H


def apply_homography(
    img_source: np.ndarray,
    img: np.ndarray,
    H: np.ndarray
) -> np.ndarray:
    """Apply homography of img to img_source.

    Args:
        img_source (np.ndarray): source image
        img (np.ndarray): image to transform
        H (np.ndarray): homography parameters

    Returns:
        np.ndarray: transformed image
    """

    # Apply homographie
    img_result = cv2.warpPerspective(
        img, H, (img_source.shape[1], img_source.shape[0]))

    # Overlay
    img_result = overlay_image_alpha(
        img_source, img_result, 0, 0)

    return img_result


def _main_(args: argparse.Namespace) -> None:
    """Main function."""

    path = os.path.join(os.path.dirname(__file__), 'src/data', args.avatar)

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
    if args.debug:
        cv2.namedWindow('Source with 3D points', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Source with 2D points', cv2.WINDOW_NORMAL)
    scale_factor = avatar_config['scale']
    shape = (np.array(avatar_config['shape']) * scale_factor).astype(np.int32)

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

        # Remove reference offset
        points *= scale_factor
        offset_ref = np.mean(points[avatar_config['reference']['points']], axis=0)
        points -= offset_ref - shape / 2

        # Draw the avatar
        img_avatar = 255 * np.ones(list(shape) + [4], dtype=np.uint8)
        for name, piece in avatar_config['pieces'].items():

            # Get current piece of the face
            img_path = os.path.join(path, name + '.png')
            avatar_piece = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            avatar_piece = cv2.resize(avatar_piece, (0, 0), fx=scale_factor, fy=scale_factor)

            # Get reference points
            ref_points = [np.array(piece['calibration'][i]) * scale_factor
                          for i in range(len(piece['calibration']))]

            for ref_points_, transfo in zip(ref_points, piece['transformations']):
                cur_points = points[transfo['points']]

                if transfo['method'] == 'transform':
                    # Get transformation parameters
                    translation, squeeze, angle = compute_transformation(
                        ref_points_ -
                        np.array([avatar_piece.shape[1] / 2,
                                  avatar_piece.shape[0] / 2]),
                        cur_points)

                    # Apply transformation
                    img_avatar = apply_transformation(
                        img_avatar,
                        avatar_piece,
                        translation,
                        squeeze,
                        angle)

                    # Update ref_points
                    transfo_mat = np.array([
                        [
                            np.cos(angle) * squeeze[0],
                            -np.sin(angle) * squeeze[1],
                            translation[0]
                        ],
                        [
                            np.sin(angle) * squeeze[0],
                            np.cos(angle) * squeeze[1],
                            translation[1]
                        ]
                    ])
                    for point_i, points_ in enumerate(ref_points):
                        points_ = np.array([
                            points_[:, 0],
                            points_[:, 1],
                            np.ones(len(points_))
                        ])
                        ref_points[point_i] = (
                            transfo_mat @ points_).transpose()

                elif transfo['method'] == 'reference':
                    # Get translation parameters
                    translation = np.mean(cur_points, axis=0) - \
                        np.mean(ref_points_, axis=0)
                    
                    # Apply translation
                    img_avatar = overlay_image_alpha(
                        img_avatar, avatar_piece, translation[0], translation[1])

                elif transfo['method'] == 'homography':
                    # Get homography matrix
                    homography = compute_homography(
                        ref_points_,
                        cur_points)

                    # Apply homography
                    img_avatar = apply_homography(
                        img_avatar,
                        avatar_piece,
                        homography)

                    # Update ref_points
                    for point_i, points_ in enumerate(ref_points):
                        points_ = np.array([
                            points_[:, 0],
                            points_[:, 1],
                            np.ones(len(points_))
                        ])
                        ref_points[point_i] = (
                            homography @ points_).transpose()[..., :2]

                elif transfo['method'] == 'squeeze':
                    # Get squeeze
                    ref_dist = (
                        np.linalg.norm(ref_points_[0] - ref_points_[1]) /
                        np.linalg.norm(ref_points_[2] - ref_points_[3])
                    )
                    cur_dist = (
                        np.linalg.norm(cur_points[0] - cur_points[1]) /
                        np.linalg.norm(cur_points[2] - cur_points[3])
                    )
                    squeeze = np.array([
                        1,
                        cur_dist / ref_dist
                    ])

                    # Save previous shape
                    prev_shape = avatar_piece.shape

                    # Apply squeeze to image and ref_points
                    avatar_piece = cv2.resize(
                        avatar_piece, (0, 0), fx=squeeze[0], fy=squeeze[1])
                    for ref_points_ in ref_points:
                        ref_points_ -= (1-squeeze) * np.array([
                            prev_shape[1] / 2,
                            prev_shape[0] / 2
                        ])

        # Display the result
        cv2.imshow('Source', frame)
        cv2.imshow('Avatar', img_avatar)
        if args.debug:

            # Draw 2D points
            frame_ = frame.copy()
            for point in points:
                cv2.circle(frame_, (int(point[0]), int(
                    point[1])), 1, (30, 80, 255), -1)
            cv2.imshow('Source with 2D points', frame_)

            # Draw 3D points
            frame_ = frame.copy()
            points = point_detector.raw_process(frame[..., :3])
            if points is not None:
                for point in points:
                    cv2.circle(frame_, (int(point[0]), int(
                        point[1])), 1, (30, 80, 255), -1)
            cv2.imshow('Source with 3D points', frame_)

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
    parser.add_argument('-d', '--debug',
                        action='store_true', help='debug mode')
    args = parser.parse_args()

    # Run main function
    _main_(args)
