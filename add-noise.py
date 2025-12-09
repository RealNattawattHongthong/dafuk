import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import face_recognition

def convert_heic_to_array(image_path):
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
        img = Image.open(image_path)
        return np.array(img.convert('RGB'))
    except ImportError:
        print("Warning: pillow-heif not installed. Skipping HEIC files.")
        return None

def load_image(image_path):
    path = Path(image_path)
    if path.suffix.upper() == '.HEIC':
        rgb_array = convert_heic_to_array(image_path)
        if rgb_array is None:
            return None
        return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    else:
        return cv2.imread(str(image_path))

def add_noise(image, noise_level=25):
    noise = np.random.normal(0, noise_level, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image
def convert_to_bw(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bw = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return bw

def create_half_face(image_path, output_path, side='left'):
    img = load_image(image_path)
    if img is None:
        return False
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_img)

    if len(face_locations) == 0:
        print(f"  No face detected, skipping half-face")
        return False

    top, right, bottom, left = face_locations[0]
    face_width = right - left
    middle = left + (face_width // 2)
    if side == 'left':

        img[:, middle:] = 0
    else:
        img[:, :middle] = 0
    cv2.imwrite(output_path, img)
    return True

def augment_dataset(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    image_extensions = ['.jpg', '.jpeg', '.png', '.heic']
    image_files = [f for f in input_path.glob('*') if f.suffix.lower() in image_extensions]

    print(f"Found {len(image_files)} images to augment")
    print(f"Output directory: {output_path}\n")

    total_augmented = 0

    for idx, image_file in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] Processing: {image_file.name}")
        img = load_image(str(image_file))
        if img is None:
            print(f"  ✗ Failed to load")
            continue
        base_name = image_file.stem
        original_output = output_path / f"{base_name}_original.jpg"
        cv2.imwrite(str(original_output), img)
        print(f"  ✓ Original saved")
        total_augmented += 1
        noisy_img = add_noise(img, noise_level=25)
        noise_output = output_path / f"{base_name}_noise.jpg"
        cv2.imwrite(str(noise_output), noisy_img)
        print(f"  ✓ Noisy version saved")
        total_augmented += 1
        bw_img = convert_to_bw(img)
        bw_output = output_path / f"{base_name}_bw.jpg"
        cv2.imwrite(str(bw_output), bw_img)
        print(f"  ✓ B&W version saved")
        total_augmented += 1
        half_left_output = output_path / f"{base_name}_half_left.jpg"
        if create_half_face(str(image_file), str(half_left_output), side='left'):
            print(f"  ✓ Half-face (left) saved")
            total_augmented += 1
        half_right_output = output_path / f"{base_name}_half_right.jpg"
        if create_half_face(str(image_file), str(half_right_output), side='right'):
            print(f"  ✓ Half-face (right) saved")
            total_augmented += 1

        print()
    print("="*60)
    print(f"Augmentation Complete!")
    print(f"Original images: {len(image_files)}")
    print(f"Augmented images: {total_augmented}")
    print(f"Total dataset size: {total_augmented}")
    print("="*60)
if __name__ == "__main__":
    INPUT_DIR = "boat dataset"
    OUTPUT_DIR = "boat dataset augmented"

    augment_dataset(INPUT_DIR, OUTPUT_DIR)
