import cv2
import numpy as np
from fpdf import FPDF
from pathlib import Path
import os

def is_duplicate(img1, img2):
    """ 두 이미지의 유사도를 비교하여 중복 여부를 판단 """
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    diff = cv2.absdiff(img1, img2)
    diff_score = np.sum(diff) / (diff.size * 255)

    return diff_score < 0.041

def crop_all_sides(image):
    """ 상하 흰색 여백만 제거 (좌우 제거는 주석처리) """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 바이너리 마스크로 흰색 아닌 부분 감지
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # === 상하 여백 제거 ===
    row_sums = np.sum(thresh, axis=1)
    top = np.argmax(row_sums > 0)
    bottom = len(row_sums) - np.argmax(row_sums[::-1] > 0)

    # === 좌우 여백 제거 (필요할 때만 사용) ===
    # col_sums = np.sum(thresh, axis=0)
    # left = np.argmax(col_sums > 0)
    # right = len(col_sums) - np.argmax(col_sums[::-1] > 0)

    # 좌우는 유지하고 상하만 자름
    cropped_img = image[top:bottom, :]

    return cropped_img


def rotate_image(image, angle):
    """ 검은 배경 없이 이미지 회전 (회전 후 여백 자동 제거) """
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)

    # 회전 매트릭스
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 새 경계 크기 계산
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # 회전 중심 이동 보정
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # 회전 수행 (배경을 흰색으로 설정)
    rotated = cv2.warpAffine(image, M, (new_w, new_h), borderValue=(255, 255, 255))
    return rotated


def capture_frames(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"🚨 Error: Unable to open video file: {video_path}")
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    ret, frame = cap.read()
    if not ret:
        print("🚨 No frames found in the video.")
        cap.release()
        return []

    frame_count = 0
    captured_images = []
    prev_cropped = None

    print("📸 캡처 시작...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 회전
        frame = rotate_image(frame, -90)
        # 상하좌우 여백 제거
        cropped_frame = crop_all_sides(frame)

        # 첫 번째 프레임은 저장
        if prev_cropped is None:
            image_path = output_dir / f"frame_{frame_count:04d}.png"
            cv2.imwrite(str(image_path), cropped_frame)
            captured_images.append(str(image_path))
            print(f"✅ 캡처됨: {image_path}")
            prev_cropped = cropped_frame
            frame_count += 1
            continue

        # 중복 체크
        if is_duplicate(prev_cropped, cropped_frame):
            print(f"🛑 중복 프레임 - 저장하지 않음: frame_{frame_count:04d}.png")
        else:
            image_path = output_dir / f"frame_{frame_count:04d}.png"
            cv2.imwrite(str(image_path), cropped_frame)
            captured_images.append(str(image_path))
            print(f"✅ 캡처됨: {image_path}")
            prev_cropped = cropped_frame

        frame_count += 1

    cap.release()
    print(f"📸 캡처 완료 - 총 {len(captured_images)} 프레임 저장됨")
    return captured_images

def images_to_pdf(image_paths, output_pdf):
    pdf = FPDF(unit="pt", format="A4")
    page_width, page_height = 595, 842
    total_images = len(image_paths)

    print(f"📄 PDF 변환 시작: 총 {total_images} 페이지")

    try:
        for idx, img_path in enumerate(image_paths):
            pdf.add_page()

            img = cv2.imread(img_path)
            if img is None:
                print(f"🚨 이미지 로드 실패: {img_path}")
                continue

            h, w = img.shape[:2]

            # 이미지 비율 계산 (PDF 페이지에 꽉 차게 배치)
            width_ratio = page_width / w
            height_ratio = page_height / h
            ratio = max(width_ratio, height_ratio)

            new_width = int(w * ratio)
            new_height = int(h * ratio)

            # 중앙 정렬
            x = (page_width - new_width) // 2
            y = (page_height - new_height) // 2

            pdf.image(img_path, x=x, y=y, w=new_width, h=new_height)

            del img  # 메모리 해제

            print(f"📄 PDF 변환 중: {idx + 1}/{total_images} 페이지 완료")

        print(f"📄 PDF 저장 중: {output_pdf}...")
        pdf.output(output_pdf)
        pdf.close()
        print(f"✅ PDF 저장 완료: {output_pdf}")

    except Exception as e:
        print(f"🚨 PDF 생성 중 에러 발생: {e}")

def main():
    video_path = "input.mp4"
    output_dir = "output_frames"
    output_pdf = "output.pdf"

    captured_images = capture_frames(video_path, output_dir)

    if captured_images:
        images_to_pdf(captured_images, output_pdf)
    else:
        print("🚨 No significant changes detected in the video")

    # 임시 이미지 삭제
    for img_path in Path(output_dir).glob("*.png"):
        img_path.unlink()
    Path(output_dir).rmdir()

if __name__ == "__main__":
    main()
