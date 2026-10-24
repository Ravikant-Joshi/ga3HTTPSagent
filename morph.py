import cv2
import mediapipe as mp
import numpy as np
import subprocess
from scipy.spatial import Delaunay
import os

def get_landmarks(image_bgr):
    mp_face_mesh = mp.solutions.face_mesh
    h, w = image_bgr.shape[:2]
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            raise ValueError("No face detected in one of the images.")
        landmarks = results.multi_face_landmarks[0].landmark
        points = [[int(pt.x * w), int(pt.y * h)] for pt in landmarks]

    # Add corner and border anchor points to anchor the entire image frame
    borders = [
        [0, 0], [w // 2, 0], [w - 1, 0],
        [0, h // 2], [w - 1, h // 2],
        [0, h - 1], [w // 2, h - 1], [w - 1, h - 1]
    ]
    points.extend(borders)
    return np.array(points, dtype=np.float32)

def apply_affine_transform(src, src_tri, dst_tri, size):
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    dst = cv2.warpAffine(
        src, warp_mat, (size[0], size[1]), None,
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )
    return dst

def morph_triangle(img1, img2, img, t1, t2, t, alpha):
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))

    t1_rect = []
    t2_rect = []
    t_rect = []

    for i in range(3):
        t_rect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))
        t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Mask for triangle
    mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t_rect), (1.0, 1.0, 1.0), 16, 0)

    # Warp patches
    img1_rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2_rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warp_img1 = apply_affine_transform(img1_rect, t1_rect, t_rect, size)
    warp_img2 = apply_affine_transform(img2_rect, t2_rect, t_rect, size)

    # Alpha blend warped patches
    img_rect = (1.0 - alpha) * warp_img1 + alpha * warp_img2

    # Place blended triangle inside the full canvas
    img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = (
        img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] * (1 - mask) + img_rect * mask
    )

def main():
    img1_path = "from.jpg"
    img2_path = "to.jpg"

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # Standardize dimensions to 480x480 for lightweight encoding
    size = (480, 480)
    img1 = cv2.resize(img1, size)
    img2 = cv2.resize(img2, size)

    points1 = get_landmarks(img1)
    points2 = get_landmarks(img2)

    # Compute Delaunay triangulation on the midpoint configuration
    mid_points = 0.5 * points1 + 0.5 * points2
    delaunay = Delaunay(mid_points)
    triangles = delaunay.simplices

    os.makedirs("frames", exist_ok=True)
    num_frames = 30  # 30 frames at 15 fps = 2.0 seconds morph

    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        morph_points = (1 - alpha) * points1 + alpha * points2
        morphed_frame = np.zeros(img1.shape, dtype=img1.dtype)

        for tri in triangles:
            t1 = points1[tri]
            t2 = points2[tri]
            t = morph_points[tri]
            morph_triangle(img1, img2, morphed_frame, t1, t2, t, alpha)

        frame_path = f"frames/frame_{i:03d}.png"
        cv2.imwrite(frame_path, morphed_frame)

    print("Frames generated. Encoding and compressing video with FFmpeg...")

    # Encode to MP4 with H.264, scaling to 360p, high CRF to guarantee < 100 KB
    cmd = [
        "ffmpeg", "-y",
        "-framerate", "15",
        "-i", "frames/frame_%03d.png",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "32",
        "-preset", "slow",
        "-vf", "scale=360:360",
        "morph.mp4"
    ]
    subprocess.run(cmd, check=True)

    file_size_bytes = os.path.getsize("morph.mp4")
    print(f"Done! Created morph.mp4 ({file_size_bytes} bytes).")
    if file_size_bytes < 100000:
        print("Success: Video size is under the 100 KB (100,000 bytes) limit.")
    else:
        print("Warning: Exceeds 100 KB. Increase -crf to 35 or lower resolution to 320:320.")

if __name__ == "__main__":
    main()