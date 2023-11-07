import cv2
import math
import mediapipe as mp
import face_part_keypoints
from PIL import Image, ImageDraw
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import List, Mapping, Optional, Tuple, Union

class faceLandmark:
    
    def __init__(self):
        self.image_file = ""
        self.world_pixel_coordinates = []
    
    def detect_landmarks(self,image_file):
        self.image_file = image_file
        self.image = mp.Image.create_from_file(image_file)
        image_rows, image_cols, _ = self.image.numpy_view().shape
        base_options = python.BaseOptions(model_asset_path='../models/face_landmarker_v2_with_blendshapes.task')
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                            output_face_blendshapes=True,
                                            output_facial_transformation_matrixes=True,
                                            num_faces=1)
        detector = vision.FaceLandmarker.create_from_options(options)
        detection_result = detector.detect(self.image)
        self.get_world_pixel_coordinates(detection_result, image_cols, image_rows)
        
        return detection_result
        
    def _normalized_to_pixel_coordinates(self,
        normalized_x: float, normalized_y: float, image_width: int,
        image_height: int) -> Union[None, Tuple[int, int]]:
        """Converts normalized value pair to pixel coordinates."""

        # Checks if the float value is between 0 and 1.
        def is_valid_normalized_value(value: float) -> bool:
            return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                            math.isclose(1, value))

        if not (is_valid_normalized_value(normalized_x) and
                is_valid_normalized_value(normalized_y)):
            # TODO: Draw coordinates even if it's outside of the image bounds.
            return None
        x_px = min(math.floor(normalized_x * image_width), image_width - 1)
        y_px = min(math.floor(normalized_y * image_height), image_height - 1)
        return x_px, y_px
    
    def get_world_pixel_coordinates(self, detection_result, image_cols, image_rows):
        for i in range(len(detection_result.face_landmarks[0])):
            self.world_pixel_coordinates.append(self._normalized_to_pixel_coordinates(detection_result.face_landmarks[0][i].x, detection_result.face_landmarks[0][i].y, image_cols, image_rows))
    
    def get_max_box(self, part_pixel_coordinate):
        min_x = float('inf')
        min_y = float('inf')
        max_x = float('-inf')
        max_y = float('-inf')
        bounding_box = []
        for x, y in part_pixel_coordinate:
            if x < min_x:
                min_x = x
            if x > max_x:
                max_x = x
            if y < min_y:
                min_y = y
            if y > max_y:
                max_y = y
        return min_x, min_y, max_x, max_y
    
    def get_part_of_face(self, face_part):
        keypoint_list = []
        if face_part == "frontal": keypoint_list = face_part_keypoints.FRONTAL
        elif face_part == "eye": keypoint_list = face_part_keypoints.EYE
        elif face_part == "nose": keypoint_list = face_part_keypoints.NOSE
        elif face_part == "lips": keypoint_list = face_part_keypoints.LIPS
        elif face_part == "leftcheeks": keypoint_list = face_part_keypoints.RIGHTCHEEKS
        elif face_part == "rightcheeks": keypoint_list = face_part_keypoints.LEFTCHEEKS
        else: keypoint_list = list(range(468))
        
        values = [self.world_pixel_coordinates[index] for index in keypoint_list]
        min_x, min_y, max_x, max_y = self.get_max_box(values)
        return (self.crop_image(min_x, min_y, max_x-min_x, max_y-min_y),values)
    
    def draw_bounding_box(image_path, x1, y1, x2, y2, color=(255, 0, 0), thickness=2):
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        # Draw the rectangle
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=thickness)
    
    def crop_image(self,x,y,width,height):
        original_image = cv2.imread(self.image_file)
        cropped_image = original_image[y:y+height, x:x+width]
        return cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)