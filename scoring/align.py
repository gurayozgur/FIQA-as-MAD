"""Align face images to 112x112 with MTCNN and the ArcFace evaluation template.

    python scoring/align.py --in /path/to/images --out /path/to/aligned --align-repo /path/to/face_alignment

Uses norm_crop of https://github.com/gurayozgur/face_alignment (createEvalDB=True). MTCNN (facenet-pytorch,
select_largest=True, post_process=False) detects on the RGB image, the landmarks of the first face are applied to the
original image, and the result is written as PNG."""
import argparse
import sys
from pathlib import Path
import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--align-repo", required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    sys.path.insert(0, a.align_repo)
    from align_trans import norm_crop
    from facenet_pytorch import MTCNN
    mtcnn = MTCNN(select_largest=True, post_process=False, device=a.device)
    src, out, failed = Path(a.src), Path(a.out), []
    for p in sorted(q for q in src.rglob("*") if q.suffix.lower() in (".jpg", ".jpeg", ".png")):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        _, _, landmarks = mtcnn.detect(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), landmarks=True)
        if landmarks is None or len(landmarks) == 0:
            failed.append(p); continue
        warped, _ = norm_crop(img, landmark=landmarks[0], image_size=112, createEvalDB=True)
        target = (out / p.relative_to(src)).with_suffix(".png")
        target.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target), warped)
    print(f"aligned: done, no face detected in {len(failed)} images", *failed[:20], sep="\n")


if __name__ == "__main__":
    main()
