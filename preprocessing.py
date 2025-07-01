import os
import cv2

# Path to your folder
folder_path = "" 

def apply_fun_preprocessing(path):
    img_bgr = cv2.imread(path)
    
    if img_bgr is None:
        print(f"Error reading image: {path}")
        return None

    img_bgr = cv2.resize(img_bgr, (224, 224))
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    v_channel = img_hsv[:, :, 2]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v_clahe = clahe.apply(v_channel)

    img_hsv[:, :, 2] = v_clahe
    img_clahe_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    
    return img_clahe_rgb


n = 0 #total number of imaages

for i in range(1, n + 1):
    image_path = os.path.join(folder_path, f"{i}.png")
    image = apply_fun_preprocessing(image_path)

    if image is None:
        continue 

    os.remove(image_path)
    cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))