import argparse

import cv2
import numpy as np
from uniface import SCRFD
from uniface.constants import SCRFDWeights

from bytetrack import BYTETracker
from utils import draw_bbox


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="Video source")
    parser.add_argument("--save", type=str, default=None, help="Output video path")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    args = parser.parse_args()

    detector = SCRFD(model_name=SCRFDWeights.SCRFD_500M_KPS)
    tracker = BYTETracker(track_thresh=0.5, track_buffer=30, match_thresh=0.8, low_thresh=0.1)

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps, (width, height))

    np.random.seed(42)
    colors = [(int(r), int(g), int(b)) for r, g, b in np.random.rand(100, 3) * 255]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces
        faces = detector.detect(frame)
        dets = np.array([[*face.bbox, face.confidence] for face in faces if face.confidence >= args.conf])
        dets = dets if len(dets) > 0 else np.empty((0, 5))

        # Update tracker
        tracks = tracker.update(dets)

        # Draw tracks
        for x1, y1, x2, y2, track_id in tracks.astype(int):
            color = colors[track_id % 100]
            draw_bbox(frame, (x1, y1, x2, y2), color=color)
            cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if writer:
            writer.write(frame)

        cv2.imshow("ByteTrack Face Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
