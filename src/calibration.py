import os
import cv2
import json
import numpy as np

from utils import *


def calibrate_avatar(path: str) -> bool:
    """Calibrate the avatar.

    Args:
        path (str): path to the avatar

    Returns:
        bool: True if calibration was successful, False otherwise
    """

    # Load avatar config file
    with open(os.path.join(path, 'config.json'), 'r') as f:
        avatar_config = json.load(f)

    # Load and stack avatar images
    img_avatar = np.zeros(avatar_config['shape'] + [4], np.uint8)
    for name, piece in avatar_config['pieces'].items():
        img_ = cv2.imread(os.path.join(
            path, f'{name}.png'), cv2.IMREAD_UNCHANGED)
        img_avatar = overlay_image_alpha(
            img_avatar, img_, piece['position'][0], piece['position'][1])

    # Initialize camera
    cam_pose = [dim / 2 for dim in img_avatar.shape[:2]]
    cam_scale = 1.0

    # Intialize tools
    cap = cv2.VideoCapture(0)
    point_detector = PointsDetector()

    # Create windows
    cv2.namedWindow('Calibration', cv2.WINDOW_NORMAL)

    calibrated = False
    running = True
    # Main loop
    while running:
        # Capture and process frame
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        points = point_detector.process(frame)

        # Skip frame if no face detected
        if points is None:
            continue

        # Draw points
        for point in points:
            cv2.circle(frame, (int(point[0]), int(
                point[1])), 1, (0, 255, 0), 2)

        # Overlay frame on avatar
        frame = np.concatenate(
            (frame, np.full((frame.shape[0], frame.shape[1], 1), 200, np.uint8)), axis=2)
        frame = cv2.resize(frame, (0, 0), fx=cam_scale, fy=cam_scale)
        img_composed = overlay_image_alpha(
            img_avatar, frame, cam_pose[0] - frame.shape[1] / 2, cam_pose[1] - frame.shape[0] / 2)

        # Draw calibration
        cv2.imshow('Calibration', img_composed)

        # Handle key presses
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:  # q or ESC
            running = False
        elif key == 81:  # left arrow
            cam_pose[0] -= 10
        elif key == 82:  # up arrow
            cam_pose[1] -= 10
        elif key == 83:  # right arrow
            cam_pose[0] += 10
        elif key == 84:  # down arrow
            cam_pose[1] += 10
        elif key == 61:  # plus
            cam_scale *= 1.1
        elif key == 45:  # minus
            cam_scale *= 0.9
        elif key == 13:  # enter

            # Compute calibrated points for each piece
            for name, piece in avatar_config['pieces'].items():
                img_ = cv2.imread(os.path.join(
                    path, f'{name}.png'), cv2.IMREAD_UNCHANGED)
                points_ = points[piece['mesh']]
                points_ *= cam_scale
                points_ += np.array([
                    cam_pose[0] - frame.shape[1] / 2,
                    cam_pose[1] - frame.shape[0] / 2
                ])
                points_ -= np.array(piece['position'])
                points_ -= np.array([
                    img_.shape[1] / 2,
                    img_.shape[0] / 2
                ])

                # Save calibration positions
                piece['calibration'] = [list(row)
                                        for row in np.round(points_, 1)]

            # Write avatar config file
            with open(os.path.join(path, 'config.json'), 'w') as f:
                json.dump(avatar_config, f, indent=4)

            calibrated = True
            running = False

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    # Print result
    if calibrated:
        print('Avatar calibrated successfully!')
    else:
        print('Avatar calibration aborted.')

    return calibrated


if __name__ == '__main__':

    # Calibrate avatar
    calibrate_avatar('data/avatar')
