#In[]
import torch
import ultralytics
from ultralytics.models.sam import Predictor as SAMPredictor
from ultralytics import SAM
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
from matplotlib.backend_tools import ToolBase
from PIL import Image
import os
import functools
import cv2
torch.cuda.get_device_name(0)


#In[]


class BoundingBoxEditor:
    def __init__(self, image_path):
        self.image = plt.imread(image_path)
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.image)
        self.rectangles = []
        self.editing = False
        
        self.RS = RectangleSelector(self.ax, self._on_rect_select, drawtype='box', useblit=True, button=[1], minspanx=5, minspany=5, spancoords='pixels', interactive=True)
        plt.connect('key_press_event', self._on_key)
        plt.show()
        
    def _on_rect_select(self, eclick, erelease):
        if not self.editing:
            x1, y1 = int(eclick.xdata), int(eclick.ydata)
            x2, y2 = int(erelease.xdata), int(erelease.ydata)
            rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
            self.ax.add_patch(rect)
            self.rectangles.append(rect)
            plt.draw()
    
    def _on_key(self, event):
        if event.key == 'e':
            if self.editing:
                self.editing = False
                plt.title("")
            else:
                self.editing = True
                plt.title("Editing: Click and drag to reshape a bounding box.")
    
    def get_bounding_boxes(self):
        return [[rect.get_x(), rect.get_y(), rect.get_width() + rect.get_x(), rect.get_height() + rect.get_y()]
                for rect in self.rectangles]

#In[]

def getBoundingBoxes(image_path):
    bbox_editor = BoundingBoxEditor(image_path)
    bounding_boxes = bbox_editor.get_bounding_boxes()
    print("Bounding Boxes:", bounding_boxes)
    return bounding_boxes


#In[]
# Function to handle mouse events
def on_mouse(event, x, y, flags, param, base_name):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if the click is within the "Skip" button area
        if x < button_width:
            cv2.destroyWindow(f"Cropped_Object_{mask_idx}")
        # Check if the click is within the "Save" button area
        elif x > button_width:
            cv2.imwrite(f"cropped_images/SAM/{base_name}_{mask_idx}.jpg", cropped_object)
            print(f"Cropped_Object_{mask_idx}.jpg saved.")
            cv2.destroyWindow(f"Cropped_Object_{mask_idx}")


# %%

overrides = dict(conf=0.25, task='segment', mode='predict', imgsz=512, model="mobile_sam.pt", device='cuda')
predictor = SAMPredictor(overrides=overrides)

image_path = "tomato.jpg"
promptable = False
prompt = 'bbox'
output_dir = "segmented masks"
os.makedirs(output_dir, exist_ok=True)

if promptable:
    if prompt == 'bbox':
        base_name, extension = os.path.splitext(image_path)
        predictor.set_image(image_path)
        bounding_boxes = getBoundingBoxes(image_path)
        results = predictor(bboxes=bounding_boxes)
        generate_mask(results)
    elif prompt == 'even_points':
        image = Image.open(image_path)
        image_width, image_height = image.size
        num_points_height = 10
        num_points_width = 10
        step_size_height = image_height // (num_points_height + 1)
        step_size_width = image_width // (num_points_width + 1)
        point_coordinates = []
        for i in range(1, num_points_height + 1):
            for j in range(1, num_points_width + 1):
                x = j * step_size_width
                y = i * step_size_height
                point_coordinates.append([x, y])
        image_np = np.array(image)
        # for point in point_coordinates:
        #     cv2.circle(image_np, point, 20, (255, 53, 200), -1)
        # image_resized = cv2.resize(image_np, (image_width // 5, image_height // 5))
        # cv2.imshow("Image with Points", image_resized)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        print(point_coordinates)
        for points in point_coordinates:
            results = predictor(points=points, labels=[1])
            generate_mask(results)
else:
    results = predictor(source=image_path, crop_n_layers=1, points_stride=16)


def generate_mask(results):

    segmentation_masks = []
    masks = results[0].masks
    for mask_idx, mask in enumerate(masks, start=1):

        mask_array = mask.data.cpu().numpy().astype(np.uint8) * 255
        # sel_image = image.copy()
        image = results[0].orig_img.copy()
        white_pixels = mask_array[0] == 255

        image[~white_pixels] = 0

        contours, _ = cv2.findContours(mask_array[0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            cropped_object = image[y:y+h, x:x+w]
            cropped_object = cv2.resize(cropped_object, (512, 512))

            button_width = 100
            button_height = 50
            cv2.namedWindow(f"Cropped_Object_{mask_idx}", cv2.WINDOW_NORMAL)

            on_mouse_with_image_path = functools.partial(on_mouse, base_name=os.path.splitext(os.path.basename(image_path))[0])
            cv2.setMouseCallback(f"Cropped_Object_{mask_idx}", on_mouse_with_image_path)

            window_height = cropped_object.shape[0] + button_height
            window_width = cropped_object.shape[1] + button_width * 2
            combined_window = np.zeros((window_height, window_width, 3), dtype=np.uint8)

            combined_window[:cropped_object.shape[0], :cropped_object.shape[1]] = cropped_object

            cv2.putText(combined_window, "Skip", (10, window_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(combined_window, "Save", (cropped_object.shape[1] + 10, window_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow(f"Cropped_Object_{mask_idx}", combined_window)

            cv2.waitKey(0)
            cv2.destroyAllWindows()



print("Segmented masks saved.")

