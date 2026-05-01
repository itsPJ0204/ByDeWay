
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# VSR spatial-relation categories (for per-category reporting)

VSR_CATEGORIES = {
    "adjacency": [
        "adjacent to", "alongside", "at the side of", "at the right side of",
        "at the left side of", "attached to", "at the back of", "ahead of",
        "against", "at the edge of",
    ],
    "directional": [
        "off", "past", "toward", "down", "away from", "along", "around",
        "into", "across", "across from", "down from",
    ],
    "orientation": [
        "facing", "facing away from", "parallel to", "perpendicular to",
    ],
    "projective": [
        "on top of", "beneath", "beside", "behind", "left of", "right of",
        "under", "in front of", "below", "above", "over", "in the middle of",
    ],
    "proximity": [
        "by", "close to", "near", "far from", "far away from",
    ],
    "topological": [
        "connected to", "detached from", "has as a part", "part of",
        "contains", "within", "at", "on", "in", "with", "surrounding",
        "among", "consists of", "out of", "between", "inside", "outside",
        "touching",
    ],
    "unallocated": [
        "beyond", "next to", "opposite to", "among", "enclosed by",
    ],
}

RELATION_TO_CATEGORY = {}
for cat, rels in VSR_CATEGORIES.items():
    for r in rels:
        RELATION_TO_CATEGORY[r] = cat


class SpatialAnalyzer:
    def __init__(self, model_path="yolov8n.pt"):
        """
        Initialize the SpatialAnalyzer with a YOLO model.
        Args:
            model_path (str): Path or name of the YOLO model to load.
                              Supports standard YOLO (e.g. yolov8n.pt) and
                              YOLO-World models (e.g. yolov8l-worldv2.pt).
        """
        print(f"Initializing SpatialAnalyzer with model='{model_path}'...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_yolo_world = "world" in model_path.lower()
        if self.is_yolo_world:
            from ultralytics import YOLOWorld
            self.model = YOLOWorld(model_path)
        else:
            self.model = YOLO(model_path)
        self._current_classes = None
        print("SpatialAnalyzer initialized.")

    def set_classes(self, classes):
        """Set target classes for YOLO-World open-vocabulary detection.

        When using a YOLO-World model, this tells the detector exactly which
        objects to look for. Includes caching to avoid redundant text-encoder
        calls when the same classes are requested consecutively.

        Args:
            classes (list[str]): List of class names to detect.
                                 Only effective when using a YOLO-World model.
        """
        if not self.is_yolo_world or not classes:
            return
        sorted_classes = sorted(set(c.strip() for c in classes if c.strip()))
        if sorted_classes == self._current_classes:
            return
        self.model.set_classes(sorted_classes)
        self._current_classes = sorted_classes

    def check_presence(self, image: np.ndarray, target_obj: str) -> bool:
        """
        Check if a specific object is present in the image.
        Used primarily for POPE hallucination benchmark.
        """
        if not target_obj:
            return False
            
        self.set_classes([target_obj])
        results = self.model.predict(image, verbose=False, device=self.device)
        result = results[0]
        
        if not result.boxes:
            return False
            
        names = result.names
        target_lower = target_obj.lower()
        
        for box in result.boxes:
            cls = int(box.cls[0])
            label = names[cls].lower()
            if target_lower in label or label in target_lower:
                return True
        return False

    def analyze(self, image_array, masks, max_relations_per_layer: int = 6):
        """
        Analyze the image and return spatial relationships between objects within the same depth layer.
        
        Args:
            image_array (np.array): Determine objects from this image (H, W, 3).
            masks (list): List of 3 boolean/binary masks [top, bottom, mid] corresponding to 
                          [Closest, Farthest, Mid Range].
                          Shape of each mask: (H, W, 1).
        
        Returns:
            list: A list of 3 strings, each containing the spatial description for the corresponding layer.
                  e.g., ["Object A is to the left of Object B...", "", ""]
        """
        # Run object detection
        results = self.model(image_array, verbose=False, stream=False)
        result = results[0]
        
        full_descriptions = ["", "", ""] # Closest, Farthest, Mid
        
        # Categorize objects into layers
        layer_objects = [[], [], []] # Closest, Farthest, Mid
        
        # Get class names
        names = result.names
        
        if not result.boxes:
             return full_descriptions

        for box in result.boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = names[cls]
            
            # Calculate center
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            # Determine which layer this object belongs to
            
            best_layer = -1
            max_overlap = -1
            
            box_area = (x2 - x1) * (y2 - y1)
            if box_area <= 0: continue

            for i, mask in enumerate(masks):
                # mask is (H, W, 1) usually boolean or 0/1
                # Extract mask crop for the box
                # Ensure coordinates are within bounds
                h, w = mask.shape[:2]
                bx1, by1 = max(0, x1), max(0, y1)
                bx2, by2 = min(w, x2), min(h, y2)
                
                if bx2 <= bx1 or by2 <= by1:
                    continue
                
                mask_crop = mask[by1:by2, bx1:bx2]
                overlap_count = np.count_nonzero(mask_crop)
                
                if overlap_count > max_overlap:
                    max_overlap = overlap_count
                    best_layer = i

            # Threshold for assignment? If max_overlap is very small, maybe ignore?
            # For now, assign to best layer if at least some overlap.
            if max_overlap > 0:
                layer_objects[best_layer].append({
                    "label": label,
                    "box": (x1, y1, x2, y2),
                    "center": (cx, cy),
                    "conf": conf
                })

        # Generate spatial descriptions for each layer
        for i in range(3):
            objs = layer_objects[i]
            if len(objs) < 2:
                continue
            
            # Add unique identifiers if multiple objects of same class exist
            # Count occurrences first
            label_counts = {}
            for obj in objs:
                label = obj["label"]
                label_counts[label] = label_counts.get(label, 0) + 1
            
            # Assign indices
            current_counts = {}
            for obj in objs:
                label = obj["label"]
                if label_counts[label] > 1:
                    idx = current_counts.get(label, 0) + 1
                    current_counts[label] = idx
                    obj["display_label"] = f"{label} {idx}"
                else:
                    obj["display_label"] = label
            
            # Sort objects from left to right to make descriptions more natural?
            # Or just pair them?
            # "Object A (left) is to the left of Object B (right)"
            
            objs.sort(key=lambda x: x["center"][0]) # Sort by X coordinate
            
            layer_desc = []
            
            for j in range(len(objs)):
                for k in range(j + 1, len(objs)):
                    obj_a = objs[j]
                    obj_b = objs[k]
                    
                    label_a = obj_a["display_label"]
                    label_b = obj_b["display_label"]
                    
                    # Logic for Left/Right
                    # Since sorted by X: A is always to the left of B (mostly)
                    # We can check vertical alignment too.
                    
                    wa = obj_a["box"][2] - obj_a["box"][0]
                    wb = obj_b["box"][2] - obj_b["box"][0]
                    ha = obj_a["box"][3] - obj_a["box"][1]
                    hb = obj_b["box"][3] - obj_b["box"][1]
                    
                    # Use a margin to decide if "Above"/"Below" or just "Left"/"Right"
                    # If vertical overlap is significant, then Left/Right.
                    # If separate vertically, then Above/Below.
                    
                    # Vertical overlap
                    y_overlap = min(obj_a["box"][3], obj_b["box"][3]) - max(obj_a["box"][1], obj_b["box"][1])
                    min_h = min(ha, hb)
                    
                    relation = ""
                    
                    # Heuristic: if y_overlap > 0.5 * min_h, they are side-by-side
                    if y_overlap > 0.5 * min_h:
                        # Side by side
                        if obj_a["center"][0] < obj_b["center"][0]:
                             relation = f"{label_a} is to the left of {label_b}"
                        else:
                             relation = f"{label_a} is to the right of {label_b}"
                    else:
                        # Above/Below
                        if obj_a["center"][1] < obj_b["center"][1]:
                            relation = f"{label_a} is above {label_b}"
                        else:
                            relation = f"{label_a} is below {label_b}"
                    
                    # Avoid duplicates or redundant info? 
                    # "A left of B", "B right of A". Just A left of B is enough.
                    
                    # Also, if multiple same-class objects, might be confusing ("cup is left of cup")
                    # We can add index or position hint?
                    # For now keep it simple as requested.
                    
                    layer_desc.append(relation)
            
            # Cap relations to avoid prompt blow-up (pairwise relations grow O(n^2)).
            if max_relations_per_layer is not None and max_relations_per_layer > 0:
                layer_desc = layer_desc[:max_relations_per_layer]
            full_descriptions[i] = ". ".join(layer_desc)
            
        return full_descriptions


    def _detect_objects(self, image_array):
        """Run YOLO and return a list of detected object dicts."""
        results = self.model(image_array, verbose=False, stream=False)
        result = results[0]
        names = result.names
        objects = []
        if not result.boxes:
            return objects
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = names[cls]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            area = (x2 - x1) * (y2 - y1)
            if area <= 0:
                continue
            objects.append({
                "label": label,
                "box": (x1, y1, x2, y2),
                "center": (cx, cy),
                "conf": conf,
                "area": area,
                "w": x2 - x1,
                "h": y2 - y1,
            })
        return objects

    @staticmethod
    def _box_iou(box_a, box_b):
        """Compute IoU between two boxes (x1,y1,x2,y2)."""
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = iw * ih
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _box_containment(outer_box, inner_box):
        """Fraction of inner_box area that is inside outer_box."""
        ax1, ay1, ax2, ay2 = outer_box
        bx1, by1, bx2, by2 = inner_box
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = iw * ih
        inner_area = (bx2 - bx1) * (by2 - by1)
        return inter / inner_area if inner_area > 0 else 0.0

    @staticmethod
    def _edge_distance(box_a, box_b):
        """Minimum distance between edges of two boxes. 0 if overlapping."""
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        dx = max(0, max(ax1 - bx2, bx1 - ax2))
        dy = max(0, max(ay1 - by2, by1 - ay2))
        return np.sqrt(dx * dx + dy * dy)

    @staticmethod
    def _center_distance(center_a, center_b):
        """Euclidean distance between two center points."""
        return np.sqrt((center_a[0] - center_b[0])**2 + (center_a[1] - center_b[1])**2)

    def _get_depth_at_object(self, depth_map, box):
        """Get mean depth value within the bounding box region."""
        x1, y1, x2, y2 = box
        h, w = depth_map.shape[:2]
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(w, x2), min(h, y2)
        if x2c <= x1c or y2c <= y1c:
            return 0.0
        region = depth_map[y1c:y2c, x1c:x2c]
        return float(np.mean(region)) if region.size > 0 else 0.0

    def analyze_vsr(self, image_array, depth_map=None, max_relations=20):
        """
        Enhanced spatial analysis that detects a rich set of spatial relations
        between detected objects. Designed for VSR benchmark evaluation.

        Args:
            image_array (np.ndarray): Input image (H, W, 3).
            depth_map (np.ndarray, optional): Depth map (H, W) with higher values = closer.
                If provided, enables depth-based relations (in front of, behind, etc.).
            max_relations (int): Maximum number of relations to return.

        Returns:
            list[dict]: List of detected relations, each dict contains:
                - "obj_a": label of first object
                - "obj_b": label of second object
                - "relation": spatial relation string (e.g., "left of")
                - "category": VSR category (e.g., "projective")
                - "confidence": float confidence score
        """
        objects = self._detect_objects(image_array)
        if len(objects) < 2:
            return []

        img_h, img_w = image_array.shape[:2]
        img_diag = np.sqrt(img_h**2 + img_w**2)

        # De-duplicate: add display labels
        label_counts = {}
        for obj in objects:
            label_counts[obj["label"]] = label_counts.get(obj["label"], 0) + 1
        current_counts = {}
        for obj in objects:
            lbl = obj["label"]
            if label_counts[lbl] > 1:
                idx = current_counts.get(lbl, 0) + 1
                current_counts[lbl] = idx
                obj["display_label"] = f"{lbl} {idx}"
            else:
                obj["display_label"] = lbl

        # Get depth values per object if depth map provided
        if depth_map is not None:
            for obj in objects:
                obj["depth"] = self._get_depth_at_object(depth_map, obj["box"])

        relations = []

        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                obj_a = objects[i]
                obj_b = objects[j]

                pair_relations = self._compute_pairwise_relations(
                    obj_a, obj_b, img_w, img_h, img_diag, depth_map is not None
                )
                relations.extend(pair_relations)

        # Sort by confidence, cap
        relations.sort(key=lambda r: r["confidence"], reverse=True)
        if max_relations > 0:
            relations = relations[:max_relations]

        return relations

    def _compute_pairwise_relations(self, obj_a, obj_b, img_w, img_h, img_diag, has_depth):
        """Compute all detectable spatial relations between two objects."""
        rels = []
        la = obj_a["display_label"]
        lb = obj_b["display_label"]
        box_a = obj_a["box"]
        box_b = obj_b["box"]
        ca = obj_a["center"]
        cb = obj_b["center"]

        # ── Projective relations ──────────────────────────────────
        # Left / Right
        x_diff = ca[0] - cb[0]
        x_threshold = min(obj_a["w"], obj_b["w"]) * 0.3
        if abs(x_diff) > x_threshold:
            if x_diff < 0:
                rels.append(self._make_rel(la, lb, "left of", "projective", 0.8))
                rels.append(self._make_rel(lb, la, "right of", "projective", 0.8))
            else:
                rels.append(self._make_rel(la, lb, "right of", "projective", 0.8))
                rels.append(self._make_rel(lb, la, "left of", "projective", 0.8))

        # Above / Below
        y_diff = ca[1] - cb[1]
        y_threshold = min(obj_a["h"], obj_b["h"]) * 0.3
        if abs(y_diff) > y_threshold:
            if y_diff < 0:
                rels.append(self._make_rel(la, lb, "above", "projective", 0.8))
                rels.append(self._make_rel(lb, la, "below", "projective", 0.8))
            else:
                rels.append(self._make_rel(la, lb, "below", "projective", 0.8))
                rels.append(self._make_rel(lb, la, "above", "projective", 0.8))

        # Beside (horizontally close, vertically aligned)
        y_overlap = min(box_a[3], box_b[3]) - max(box_a[1], box_b[1])
        min_h = min(obj_a["h"], obj_b["h"])
        if y_overlap > 0.4 * min_h:
            edge_dist = self._edge_distance(box_a, box_b)
            if edge_dist < 0.15 * img_diag:
                rels.append(self._make_rel(la, lb, "beside", "projective", 0.7))
                rels.append(self._make_rel(lb, la, "beside", "projective", 0.7))

        # On top of / beneath (A is above B AND they overlap/touch vertically)
        if y_diff < -y_threshold:
            v_gap = box_b[1] - box_a[3]  # gap between bottom of A and top of B
            if v_gap < min_h * 0.3:  # touching or slight overlap
                x_overlap = min(box_a[2], box_b[2]) - max(box_a[0], box_b[0])
                min_w = min(obj_a["w"], obj_b["w"])
                if x_overlap > 0.3 * min_w:
                    rels.append(self._make_rel(la, lb, "on top of", "projective", 0.75))
                    rels.append(self._make_rel(lb, la, "beneath", "projective", 0.75))
                    rels.append(self._make_rel(la, lb, "over", "projective", 0.7))
                    rels.append(self._make_rel(lb, la, "under", "projective", 0.7))
        elif y_diff > y_threshold:
            v_gap = box_a[1] - box_b[3]
            if v_gap < min_h * 0.3:
                x_overlap = min(box_a[2], box_b[2]) - max(box_a[0], box_b[0])
                min_w = min(obj_a["w"], obj_b["w"])
                if x_overlap > 0.3 * min_w:
                    rels.append(self._make_rel(lb, la, "on top of", "projective", 0.75))
                    rels.append(self._make_rel(la, lb, "beneath", "projective", 0.75))
                    rels.append(self._make_rel(lb, la, "over", "projective", 0.7))
                    rels.append(self._make_rel(la, lb, "under", "projective", 0.7))

        # In the middle of (A center is within B box AND A is much smaller)
        if (box_b[0] < ca[0] < box_b[2] and box_b[1] < ca[1] < box_b[3]
                and obj_a["area"] < 0.5 * obj_b["area"]):
            rels.append(self._make_rel(la, lb, "in the middle of", "projective", 0.65))

        # In front of / behind (depth-based)
        if has_depth and "depth" in obj_a and "depth" in obj_b:
            depth_diff = obj_a["depth"] - obj_b["depth"]
            # Higher depth = closer to camera = "in front of"
            depth_threshold = 5.0  # configurable threshold
            if abs(depth_diff) > depth_threshold:
                if depth_diff > 0:
                    rels.append(self._make_rel(la, lb, "in front of", "projective", 0.7))
                    rels.append(self._make_rel(lb, la, "behind", "projective", 0.7))
                else:
                    rels.append(self._make_rel(lb, la, "in front of", "projective", 0.7))
                    rels.append(self._make_rel(la, lb, "behind", "projective", 0.7))

        # ── Proximity relations ───────────────────────────────────
        center_dist = self._center_distance(ca, cb)
        norm_dist = center_dist / img_diag  # 0..~1

        if norm_dist < 0.15:
            rels.append(self._make_rel(la, lb, "near", "proximity", 0.75))
            rels.append(self._make_rel(lb, la, "near", "proximity", 0.75))
            rels.append(self._make_rel(la, lb, "close to", "proximity", 0.7))
            rels.append(self._make_rel(lb, la, "close to", "proximity", 0.7))
            rels.append(self._make_rel(la, lb, "by", "proximity", 0.65))
            rels.append(self._make_rel(lb, la, "by", "proximity", 0.65))
        elif norm_dist > 0.5:
            rels.append(self._make_rel(la, lb, "far from", "proximity", 0.7))
            rels.append(self._make_rel(lb, la, "far from", "proximity", 0.7))
            rels.append(self._make_rel(la, lb, "far away from", "proximity", 0.65))
            rels.append(self._make_rel(lb, la, "far away from", "proximity", 0.65))

        # ── Adjacency relations ───────────────────────────────────
        edge_dist = self._edge_distance(box_a, box_b)
        norm_edge_dist = edge_dist / img_diag

        if norm_edge_dist < 0.03:  # very close edges (touching or nearly touching)
            rels.append(self._make_rel(la, lb, "adjacent to", "adjacency", 0.75))
            rels.append(self._make_rel(lb, la, "adjacent to", "adjacency", 0.75))
            rels.append(self._make_rel(la, lb, "alongside", "adjacency", 0.7))
            rels.append(self._make_rel(lb, la, "alongside", "adjacency", 0.7))
            rels.append(self._make_rel(la, lb, "at the side of", "adjacency", 0.7))
            rels.append(self._make_rel(lb, la, "at the side of", "adjacency", 0.7))

            # at the right/left side of
            if ca[0] < cb[0]:
                rels.append(self._make_rel(la, lb, "at the left side of", "adjacency", 0.65))
                rels.append(self._make_rel(lb, la, "at the right side of", "adjacency", 0.65))
            else:
                rels.append(self._make_rel(la, lb, "at the right side of", "adjacency", 0.65))
                rels.append(self._make_rel(lb, la, "at the left side of", "adjacency", 0.65))

        if edge_dist < 3:  # touching (edges overlap or within 3px)
            rels.append(self._make_rel(la, lb, "touching", "topological", 0.7))
            rels.append(self._make_rel(lb, la, "touching", "topological", 0.7))
            rels.append(self._make_rel(la, lb, "against", "adjacency", 0.65))
            rels.append(self._make_rel(lb, la, "against", "adjacency", 0.65))
            rels.append(self._make_rel(la, lb, "attached to", "adjacency", 0.6))
            rels.append(self._make_rel(lb, la, "attached to", "adjacency", 0.6))

        # Ahead of / at the back of (depth-based adjacency)
        if has_depth and "depth" in obj_a and "depth" in obj_b:
            depth_diff = obj_a["depth"] - obj_b["depth"]
            if abs(depth_diff) > depth_threshold:
                if depth_diff > 0:
                    rels.append(self._make_rel(la, lb, "ahead of", "adjacency", 0.6))
                    rels.append(self._make_rel(lb, la, "at the back of", "adjacency", 0.6))
                else:
                    rels.append(self._make_rel(lb, la, "ahead of", "adjacency", 0.6))
                    rels.append(self._make_rel(la, lb, "at the back of", "adjacency", 0.6))

        # ── Topological relations ─────────────────────────────────
        iou = self._box_iou(box_a, box_b)

        # Contains / inside / within
        containment_a_in_b = self._box_containment(box_b, box_a)
        containment_b_in_a = self._box_containment(box_a, box_b)

        if containment_a_in_b > 0.8:
            rels.append(self._make_rel(la, lb, "inside", "topological", 0.75))
            rels.append(self._make_rel(la, lb, "within", "topological", 0.7))
            rels.append(self._make_rel(la, lb, "in", "topological", 0.65))
            rels.append(self._make_rel(lb, la, "contains", "topological", 0.7))
            rels.append(self._make_rel(lb, la, "surrounding", "topological", 0.6))
        if containment_b_in_a > 0.8:
            rels.append(self._make_rel(lb, la, "inside", "topological", 0.75))
            rels.append(self._make_rel(lb, la, "within", "topological", 0.7))
            rels.append(self._make_rel(lb, la, "in", "topological", 0.65))
            rels.append(self._make_rel(la, lb, "contains", "topological", 0.7))
            rels.append(self._make_rel(la, lb, "surrounding", "topological", 0.6))

        # On (A above B center, significant overlap, common for objects resting on surfaces)
        if (ca[1] < cb[1] and iou > 0.05 and y_overlap > 0
                and obj_a["area"] < obj_b["area"] * 2):
            rels.append(self._make_rel(la, lb, "on", "topological", 0.6))

        # at the edge of: A is partially overlapping B's boundary region
        if 0.05 < containment_a_in_b < 0.75 and edge_dist < 0.05 * img_diag:
            rels.append(self._make_rel(la, lb, "at the edge of", "adjacency", 0.6))
        if 0.05 < containment_b_in_a < 0.75 and edge_dist < 0.05 * img_diag:
            rels.append(self._make_rel(lb, la, "at the edge of", "adjacency", 0.6))

        # connected to — objects are touching or overlapping
        if edge_dist < 3 or iou > 0.01:
            rels.append(self._make_rel(la, lb, "connected to", "topological", 0.6))
            rels.append(self._make_rel(lb, la, "connected to", "topological", 0.6))

        # detached from — objects are clearly separated
        if edge_dist > 0.1 * img_diag and iou < 0.01:
            rels.append(self._make_rel(la, lb, "detached from", "topological", 0.6))
            rels.append(self._make_rel(lb, la, "detached from", "topological", 0.6))

        # has as a part / consists of — one largely contains the other (relaxed)
        if containment_b_in_a > 0.6:
            rels.append(self._make_rel(la, lb, "has as a part", "topological", 0.55))
            rels.append(self._make_rel(la, lb, "consists of", "topological", 0.5))
        if containment_a_in_b > 0.6:
            rels.append(self._make_rel(lb, la, "has as a part", "topological", 0.55))
            rels.append(self._make_rel(lb, la, "consists of", "topological", 0.5))

        # part of — A is contained within B
        if containment_a_in_b > 0.6:
            rels.append(self._make_rel(la, lb, "part of", "topological", 0.55))
        if containment_b_in_a > 0.6:
            rels.append(self._make_rel(lb, la, "part of", "topological", 0.55))

        # outside — A is clearly not inside B (with proximity to avoid noise)
        if containment_a_in_b < 0.1 and norm_dist < 0.4:
            rels.append(self._make_rel(la, lb, "outside", "topological", 0.55))
        if containment_b_in_a < 0.1 and norm_dist < 0.4:
            rels.append(self._make_rel(lb, la, "outside", "topological", 0.55))

        # out of — A is just outside B (near but not inside)
        if containment_a_in_b < 0.15 and edge_dist < 0.05 * img_diag:
            rels.append(self._make_rel(la, lb, "out of", "topological", 0.5))
        if containment_b_in_a < 0.15 and edge_dist < 0.05 * img_diag:
            rels.append(self._make_rel(lb, la, "out of", "topological", 0.5))

        # at — general location association (close or overlapping)
        if norm_dist < 0.2 or iou > 0.05:
            rels.append(self._make_rel(la, lb, "at", "topological", 0.5))
            rels.append(self._make_rel(lb, la, "at", "topological", 0.5))

        # with — general co-presence (objects in same area)
        if norm_dist < 0.3:
            rels.append(self._make_rel(la, lb, "with", "topological", 0.5))
            rels.append(self._make_rel(lb, la, "with", "topological", 0.5))

        # among — A is partially within B's area
        if 0.15 < containment_a_in_b < 0.75 and iou > 0.05:
            rels.append(self._make_rel(la, lb, "among", "topological", 0.45))
        if 0.15 < containment_b_in_a < 0.75 and iou > 0.05:
            rels.append(self._make_rel(lb, la, "among", "topological", 0.45))

        # between — A is positioned within B's spatial extent (approximate)
        if 0.3 < containment_a_in_b < 0.7:
            rels.append(self._make_rel(la, lb, "between", "topological", 0.45))
        if 0.3 < containment_b_in_a < 0.7:
            rels.append(self._make_rel(lb, la, "between", "topological", 0.45))

        # Next to (unallocated — general proximity with some offset)
        if 0.05 < norm_dist < 0.3:
            rels.append(self._make_rel(la, lb, "next to", "unallocated", 0.65))
            rels.append(self._make_rel(lb, la, "next to", "unallocated", 0.65))

        return rels

    def analyze_vsr_for_caption(self, image_array, depth_map=None):
        """
        Generate a rich spatial description string from detected relations.
        Suitable for injecting into VLM prompts as context.

        Returns:
            str: Human-readable spatial relations description.
        """
        relations = self.analyze_vsr(image_array, depth_map=depth_map, max_relations=15)
        if not relations:
            return ""

        # Group by category for a cleaner description
        by_cat = {}
        for rel in relations:
            cat = rel["category"]
            if cat not in by_cat:
                by_cat[cat] = []
            desc = f"{rel['obj_a']} is {rel['relation']} {rel['obj_b']}"
            if desc not in by_cat[cat]:
                by_cat[cat].append(desc)

        parts = []
        for cat in ["projective", "proximity", "adjacency", "topological", "orientation", "directional", "unallocated"]:
            if cat in by_cat:
                # Take top few per category to avoid bloating
                cat_rels = by_cat[cat][:4]
                parts.extend(cat_rels)

        return ". ".join(parts) if parts else ""

    @staticmethod
    def _make_rel(obj_a, obj_b, relation, category, confidence):
        """Create a relation dict."""
        return {
            "obj_a": obj_a,
            "obj_b": obj_b,
            "relation": relation,
            "category": category,
            "confidence": confidence,
        }
