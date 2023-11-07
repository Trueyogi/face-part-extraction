import sys
sys.path.append("face_part_extract")
from face_part_extract.FaceLandmark import faceLandmark
import joblib
import cv2
from skimage.feature import hog
import argparse
import matplotlib.pyplot as plt

def extract_hog_features(image):
    # Convert the image to grayscale (if it's not already)
    if len(image.shape) > 2:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
        
    gray_image = cv2.resize(gray_image, (374, 304))

    # Define HOG parameters
    orientations = 9
    pixels_per_cell = (8, 8)
    cells_per_block = (2, 2)

    # Extract HOG features
    features, hog_image = hog(gray_image, orientations=orientations,
                              pixels_per_cell=pixels_per_cell,
                              cells_per_block=cells_per_block,
                              visualize=True, block_norm='L2-Hys')

    return features, hog_image

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to the image file.")
    parser.add_argument("model_path", help="Path to the model file.")
    parser.add_argument("--threshold", help="Probability threshold for wrinkle classification. Default value is 0.26.")
    parser.add_argument("--save_regions", action="store_true", help="Save the extracted face regions.")
    args = parser.parse_args()
    
    model = joblib.load(args.model_path)
    
    if args.threshold:
        THRESHOLD = args.threshold
    else:
        THRESHOLD = 0.26
    
    fl = faceLandmark()
    fl.detect_landmarks(args.image_path)

    regions = ["frontal", "eye", "nose", "lips", "leftcheeks", "rightcheeks"]
    region_images = []
    plt.figure(figsize=(12, 8))
    
    for i, region in enumerate(regions):
        region_image = fl.get_part_of_face(region)[0]
        region_images.append((region, region_image))
        
        if args.save_regions:
            plt.subplot(2, 3, i+1)
            plt.xticks([])
            plt.yticks([])
            plt.title(region)
            plt.imshow(region_image)
        
    results = []

    for region_name, image in region_images:
        features, _ = extract_hog_features(image)
        features = features[None]
        probability = model.predict_proba(features)[:, 1]
        has_wrinkles = (probability > THRESHOLD)
        results.append((region_name, probability, has_wrinkles))
        
    for result in results:
        decision = ""
        if result[2]:
            decision = "has"
        else:
            decision = "has no"
            
        print(f"The region {result[0]} {decision} wrinkles. (Probability: {result[1]}.)")
        
    if args.save_regions:
        plt.savefig("regions.png")
        
if __name__ == "__main__":
    main()