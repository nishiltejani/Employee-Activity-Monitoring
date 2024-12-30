# from ultralytics import YOLO

# class Classifier:
#     def __init__(self, model_paths: dict):
#         self.models = {key: YOLO(path) for key, path in model_paths.items()}
            
#     def classify(self, bboxes, image):
#         categories = []
#         for x1, y1, x2, y2, *_ in bboxes:
#             if x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0]:
#                 categories.append([])
#                 continue
#             tp = image[y1:y2, x1:x2]
#             sub_result = [
#                 self.models[model](tp, verbose=False)[0].names[
#                     self.models[model](tp, verbose=False)[0].probs.top1
#                 ]
#                 for model in self.models
#             ]
#             categories.append(sub_result)
#         return categories


from ultralytics import YOLO
import torch

class Classifier:
    def __init__(self, model_paths: dict, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load models and move them to the specified device
        self.models = {key: YOLO(path).to(self.device) for key, path in model_paths.items()}
            
    def classify(self, bboxes, image):
        categories = []
        for x1, y1, x2, y2, *_ in bboxes:
            if x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0]:
                categories.append([])
                continue
            tp = image[y1:y2, x1:x2]
            sub_result = []
            for model in self.models:
                try:
                    results = self.models[model](tp, verbose=False)
                    label = results[0].names[results[0].probs.top1]
                    sub_result.append(label)
                except Exception as e:
                    print(f"Error during classification with model {model}: {e}")
                    sub_result.append("unknown")
            categories.append(sub_result)
        return categories
