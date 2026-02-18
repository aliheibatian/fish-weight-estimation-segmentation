# import cv2
# import numpy as np
# from segment_anything import SamPredictor, sam_model_registry
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import torch
# import gc
# import os
# import csv

# # آزادسازی حافظه GPU
# torch.cuda.empty_cache()
# gc.collect()

# # متغیرهای سراسری
# points = []
# img = None
# pixel_to_cm = 0.0529 # مقیاس پیش‌فرض، بعداً باید کالیبره شود
# weights = []
# lengths = []
# areas = []

# def segment_fish(image, x, y, fish_idx):
#     global weights, lengths, areas, pixel_to_cm
#     try:
#         print(f"Starting segmentation for fish {fish_idx} at point ({x}, {y})...")
#         checkpoint_path = "/home/ali/Project/fish_segmentation_SAM/checkpoints/sam_vit_h_4b8939.pth"
#         if not os.path.exists(checkpoint_path):
#             print(f"Error: Model checkpoint not found at {checkpoint_path}")
#             exit()
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         print(f"Using device: {device}")
#         sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path).to(device)
#         predictor = SamPredictor(sam)

#         # تغییر اندازه تصویر به رزولوشن پیشنهادی مقاله
#         image = cv2.resize(image, (520, 390))
#         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#         print("Setting image for predictor...")
#         predictor.set_image(image_rgb)

#         # استفاده از چند نقطه برای بهبود تقسیم‌بندی
#         input_point = np.array([[x, y], [x+50, y], [x-50, y], [x, y+50], [x, y-50]])
#         input_label = np.array([1, 1, 1, 1, 1])

#         print("Running SAM prediction...")
#         masks, scores, _ = predictor.predict(
#             point_coords=input_point,
#             point_labels=input_label,
#             multimask_output=True,
#         )

#         # انتخاب بهترین ماسک
#         best_mask = masks[np.argmax(scores)]

#         # محاسبه مساحت
#         area_pixels = np.sum(best_mask)
#         area_cm2 = area_pixels * (pixel_to_cm ** 2)
#         if area_cm2 < 10 or area_cm2 > 5000:
#             print(f"Warning: Calculated area for fish {fish_idx} might be unrealistic: {area_cm2:.2f} cm²")

#         # محاسبه طول
#         contours, _ = cv2.findContours(best_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         length_cm = 0
#         if contours:
#             contour = max(contours, key=cv2.contourArea)
#             (x, y), (w, h), angle = cv2.fitEllipse(contour)
#             length_cm = max(w, h) * pixel_to_cm

#         # تخمین وزن با فرمول غیرخطی (بر اساس مقاله)
#         a, b = 0.01, 1.5  # مقادیر تقریبی، باید کالیبره شوند
#         weight = a * (area_cm2 ** b)
#         print(f"Fish {fish_idx} Area: {area_cm2:.2f} cm², Length: {length_cm:.2f} cm, Weight: {weight:.2f} g")
#         weights.append(weight)
#         lengths.append(length_cm)
#         areas.append(area_cm2)

#         # ذخیره تصویر خروجی
#         plt.figure(figsize=(10, 5))
#         plt.subplot(1, 2, 1)
#         plt.title("Original Image")
#         plt.imshow(image_rgb)
#         plt.axis("off")
#         plt.subplot(1, 2, 2)
#         plt.title(f"Fish {fish_idx} (Area: {area_cm2:.2f} cm², Length: {length_cm:.2f} cm, Weight: {weight:.2f} g)")
#         plt.imshow(image_rgb)
#         plt.imshow(best_mask, cmap="jet", alpha=0.5)
#         plt.axis("off")
#         plt.savefig(f"output_fish_{fish_idx}.png")
#         plt.close()

#         cv2.imwrite(f"segmented_fish_{fish_idx}.png", (best_mask * 255).astype(np.uint8))
#         print(f"Output saved: output_fish_{fish_idx}.png, segmented_fish_{fish_idx}.png")

#     except Exception as e:
#         print(f"Error in segmentation: {e}")
#     finally:
#         print("Cleaning up...")
#         predictor = None
#         sam = None
#         torch.cuda.empty_cache()
#         gc.collect()
#         if torch.cuda.is_available():
#             torch.cuda.synchronize()

# def click_event(event, x, y, flags, param):
#     global points, img
#     if event == cv2.EVENT_LBUTTONDOWN:
#         points.append((x, y))
#         cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
#         cv2.imshow("image", img)

#         fish_idx = len(points)
#         segment_fish(img.copy(), x, y, fish_idx)

#         cv2.imshow("image", img)

# def main():
#     global img, points
#     img = cv2.imread("1.jpg")
#     if img is None:
#         print("Error: Could not load image. Please provide a valid image file (e.g., 1.jpeg).")
#         exit()

#     print("Click on each fish to segment. Press 'q' to finish.")
#     cv2.imshow("image", img)
#     cv2.setMouseCallback("image", click_event)
#     while True:
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q'):
#             break
#     cv2.destroyAllWindows()
#     cv2.waitKey(1)

#     # ذخیره نتایج در فایل CSV
#     print("\nSummary of results:")
#     with open("fish_weights.csv", "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["Fish ID", "Area (cm²)", "Length (cm)", "Weight (g)"])
#         for idx, (area, length, weight) in enumerate(zip(areas, lengths, weights)):
#             print(f"Fish {idx+1}: Area: {area:.2f} cm², Length: {length:.2f} cm, Weight: {weight:.2f} g")
#             writer.writerow([idx+1, f"{area:.2f}", f"{length:.2f}", f"{weight:.2f}"])

# if __name__ == "__main__":
#     main()








import cv2
import numpy as np
from segment_anything import SamPredictor, sam_model_registry
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import gc
import os
import csv

torch.cuda.empty_cache()
gc.collect()    

points = []
img = None
pixel_to_cm = 0.04 
weights = []
lengths = []
areas = []
predictor = None 

def initialize_sam_model():
    global predictor
    try:
        checkpoint_path = "/home/ali/Project/fish_segmentation_SAM/checkpoints/sam_vit_h_4b8939.pth"
        if not os.path.exists(checkpoint_path):
            print(f"Error: Model checkpoint not found at {checkpoint_path}")
            exit()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path).to(device)
        predictor = SamPredictor(sam)
        return predictor
    except Exception as e:
        print(f"Error initializing SAM model: {e}")
        exit()

def segment_fish(image, x, y, fish_idx):
    global weights, lengths, areas, pixel_to_cm, predictor
    try:
        print(f"Starting segmentation for fish {fish_idx} at point ({x}, {y})...")
        
        image = cv2.resize(image, (1024, 1024))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if predictor is None:
            predictor = initialize_sam_model()
        print("Setting image for predictor...")
        predictor.set_image(image_rgb)

        scale_x = 1024 / img.shape[1]
        scale_y = 1024 / img.shape[0]
        scaled_x, scaled_y = x * scale_x, y * scale_y

        input_point = np.array([
            [scaled_x, scaled_y],
            [scaled_x + 20 * scale_x, scaled_y],
            [scaled_x - 20 * scale_x, scaled_y],
            [scaled_x, scaled_y + 20 * scale_y],
            [scaled_x, scaled_y - 20 * scale_y]
        ])
        input_label = np.array([1, 1, 1, 1, 1])

        print("Running SAM prediction...")
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        best_mask_idx = np.argmax(scores)
        best_mask = masks[best_mask_idx]
        best_score = scores[best_mask_idx]
        print(f"Best mask score: {best_score:.3f}")

        if best_score < 0.85:
            print(f"Warning: Low confidence score for fish {fish_idx}: {best_score:.3f}")
            return

        area_pixels = np.sum(best_mask)
        area_cm2 = area_pixels * (pixel_to_cm ** 2)
        if area_cm2 < 10 or area_cm2 > 5000:
            print(f"Warning: Calculated area for fish {fish_idx} might be unrealistic: {area_cm2:.2f} cm²")
            return

        contours, _ = cv2.findContours(best_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        length_cm = 0
        if contours:
            contour = max(contours, key=cv2.contourArea)
            (x, y), (w, h), angle = cv2.fitEllipse(contour)
            length_cm = max(w, h) * pixel_to_cm

        a, b = 0.01, 1.5
        weight = a * (area_cm2 ** b)
        print(f"Fish {fish_idx} Area: {area_cm2:.2f} cm², Length: {length_cm:.2f} cm, Weight: {weight:.2f} g")
        weights.append(weight)
        lengths.append(length_cm)
        areas.append(area_cm2)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(image_rgb)
        plt.plot(input_point[:, 0], input_point[:, 1], 'ro')
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.title(f"Fish {fish_idx} (Area: {area_cm2:.2f} cm², Length: {length_cm:.2f} cm, Weight: {weight:.2f} g)")
        plt.imshow(image_rgb)
        plt.imshow(best_mask, cmap="jet", alpha=0.5)
        plt.axis("off")
        plt.savefig(f"output_fish_{fish_idx}.png", bbox_inches='tight')
        plt.close()

        mask_resized = cv2.resize(best_mask.astype(np.uint8), (img.shape[1], img.shape[0]))
        cv2.imwrite(f"segmented_fish_{fish_idx}.png", mask_resized * 255)

        print(f"Output saved: output_fish_{fish_idx}.png, segmented_fish_{fish_idx}.png")

    except Exception as e:
        print(f"Error in segmentation: {e}")
    finally:
        print("Cleaning up memory...")
        torch.cuda.empty_cache()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

def click_event(event, x, y, flags, param):
    global points, img
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("image", img)

        fish_idx = len(points)
        segment_fish(img.copy(), x, y, fish_idx)

def main():
    global img, predictor
    img = cv2.imread("1.jpg")
    if img is None:
        print("Error: Could not load image. Please provide a valid image file (e.g., 1.jpg).")
        exit()

    print("Click on each fish to segment. Press 'q' to finish.")
    cv2.imshow("image", img)
    cv2.setMouseCallback("image", click_event)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    predictor = None
    torch.cuda.empty_cache()
    gc.collect()

    print("\nSummary of results:")
    with open("fish_weights.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Fish ID", "Area (cm²)", "Length (cm)", "Weight (g)"])
        for idx, (area, length, weight) in enumerate(zip(areas, lengths, weights)):
            print(f"Fish {idx+1}: Area: {area:.2f} cm², Length: {length:.2f} cm, Weight: {weight:.2f} g")
            writer.writerow([idx+1, f"{area:.2f}", f"{length:.2f}", f"{weight:.2f}"])

if __name__ == "__main__":
    main()