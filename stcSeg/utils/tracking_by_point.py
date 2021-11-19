import os
import sys
import numpy as np

sys.path.append('../mots_tools')
import pycocotools.mask as rletools

from detectron2.utils.colormap import random_color


class TrackingInfo(object):
    def __init__(self, x, y, z=None, mask=None, color=None, track_id=None, survival=2):  # destroyed after 2 frame
        if z is not None:
            self.d = x
            self.h = y

        else:
            self.x = x
            self.y = y 

        self.z = z

        self.survival = survival
        self.found = True

        self.mask = mask
        self.color = color
        self.track_id = track_id

    def update(self, x, y, z=None, mask=None, survival=2):
        if z is not None:
            self.d = x
            self.h = y

        else:
            self.x = x
            self.y = y 

        self.z = z

        self.survival = survival
        self.found = True

        self.mask = mask

    def forward_a_step(self):
        self.survival -= 1
        self.found = False
        
        return self.survival

    def get_data(self):
        if self.z is not None:
            return self.d, self.h, self.z
        else:
            return self.x, self.y

    def get_color(self):
        return self.color

    def get_mask(self):
        return self.mask

    def get_id(self):
        return self.track_id

    def is_found(self):
        return self.found

    def is_survival(self):
        return self.survival >= 0


# def to_objects(masks, args, id_divisor=1000):
#     if len(masks) == 0:
#         return [], []
    
#     h = masks[0].shape[0]
#     w = masks[0].shape[1]

#     # to solve overlapped masks

#     all_masked = np.zeros((h, w), dtype=np.uint8)

#     obj_list = []
#     new_masks = []

#     for instance_id, old_mask in enumerate(masks):
#         # print(old_mask.shape)
#         count = 0
#         mask = np.zeros((h, w), dtype=np.uint8)
#         for i, line in enumerate(old_mask):
#             for j, value in enumerate(line):
#                 # print(values)
#                 if value > 0 and all_masked[i, j] == 0:
#                     mask[i, j] = 1
#                     count += 1

#         if count < args.min_mask_area:
#             continue

#         all_masked += mask
#         mask = np.asfortranarray(mask)
#         obj = SegmentedObject(rletools.encode(mask), 1, 1 * id_divisor + instance_id)
#         obj_list.append(obj)
#         new_masks.append(mask)

#     return obj_list, new_masks



def track_objects(predictions, tracking_dict, mask_tracking_distance=300):

    masks = np.asarray(predictions.pred_masks)
    # print(masks.shape)
    classes = predictions.pred_classes
    # print(classes)
    boxes = predictions.pred_boxes


    for class_id in tracking_dict:
        for track_id in tracking_dict[class_id]:

            tracking_dict[class_id][track_id].forward_a_step()


    this_positions = []

    for box in boxes:

        x, y, w, h = box[0], box[1], box[2], box[3]
        c_x = int(x + w / 2)
        c_y = int(y + h / 2)

        this_positions.append((c_x, c_y))

    colors = 'bgrcmyk'

    color_bgr = {'b': [255, 0, 0], 'g': [0, 255, 0], 'r': [0, 0, 255], 
                 'c': [255, 255, 10], 'm': [255, 0, 255], 'y': [0, 255, 255],
                 'k': [0, 0, 0]}

    previous_matching_maps = {}

    # found the min d for objects in obj_list
    for class_id in tracking_dict:

        previous_matching_maps[class_id] = {}

        # found the min d for objects in tracking_dict
        for track_id in tracking_dict[class_id]:
            previous_obj = tracking_dict[class_id][track_id]

            if not previous_obj.is_survival():
                continue

            min_d = mask_tracking_distance ** 2  # float("inf")
            matching_id = None

            pr_x, pr_y = previous_obj.get_data()
            for this_id, (x, y) in enumerate(this_positions):
                this_class_id = classes[this_id]
                # To Do: more functions!!!!!!!!!!!!
                d = (pr_x - x) ** 2 + (pr_y - y) ** 2
                if d < min_d:
                    min_d = d
                    matching_id = this_id

            previous_matching_maps[class_id][track_id] = matching_id

    track_id_list = []
    assigned_colors = []

    for this_id, (x, y) in enumerate(this_positions):

        class_id = int(classes[this_id])

        min_d = mask_tracking_distance ** 2 # float("inf")
        found_id = None

        # print(tracking_dict)

        if class_id not in tracking_dict.keys():
            tracking_dict[class_id] = {}


        for track_id in tracking_dict[class_id]:
            previous_obj = tracking_dict[class_id][track_id]

            # To Do: more stategies!!!!!!!!!!!!
            if previous_obj.is_found() or not previous_obj.is_survival():
                continue

            pr_x, pr_y = previous_obj.get_data()

            # To Do: more functions!!!!!!!!!!!!
            d = (pr_x - x) ** 2 + (pr_y - y) ** 2 

            if d < min_d:
                min_d = d
                found_id = track_id

        if found_id is not None and previous_matching_maps[class_id][found_id] == this_id:
            tracking_dict[class_id][found_id].update(x, y, mask=masks[this_id])
        else:
            if len(tracking_dict[class_id].keys()) != 0:
                new_track_id = max(tracking_dict[class_id].keys()) + 1
            else:
                new_track_id = 0

            # color = color_bgr[colors[(new_track_id) % len(colors)]]
            color = random_color(rgb=True, maximum=1)

            tracking_dict[class_id][new_track_id] = TrackingInfo(x, y, mask=masks[this_id], color=color, track_id=new_track_id)
            found_id = new_track_id

        # print(found_id)
        color = tracking_dict[class_id][found_id].get_color()

        track_id_list.append(found_id)
        assigned_colors.append(color)





    return tracking_dict, track_id_list, assigned_colors


def convert_class_id(class_id, class_names, dataset="MOTS KITTI"):

    assert class_names is not None

    if dataset == "MOTS KITTI":
        class_dict = {"car": 1, "pedestrian": 2}

    # print(class_names)

    if class_names[class_id] in ["car", "Car"]:
        # COCO: car
        # KITTI OBJECT: Car
        class_id = class_dict["car"]

    elif class_names[class_id] in ["person", "Pedestrian"]:
        # COCO: person
        # KITTI OBJECT: Pedestrian, (ignore: Cyclist, Person_sitting)
        class_id = class_dict["pedestrian"]

    else:
        return None

    return class_id


def remove_overlap(masks):
    if len(masks) == 0:
        return []
    
    h = masks[0].shape[0]
    w = masks[0].shape[1]

    # to solve overlapped masks

    all_masked = np.zeros((h, w))

    new_masks = []

    for old_mask in masks:
        
        mask = old_mask - all_masked
        mask = np.maximum(mask, 0)
        all_masked += mask

        # for line in mask:
        #     for value in line:
        #         if value < 0:
        #             print(value)

        mask = mask.astype(np.uint8)
        mask = np.asfortranarray(mask)
        new_masks.append(mask)

    return new_masks



def save_instance(tracking_dict, path, class_names, instance_txt_out_dir, to_remove_overlap=True, id_divisor=1000):

    # print(class_names)

    img_name = path.split('/')[-1].split('.')[0]
    t = int(img_name)

    image_dir = path.split('/')[-2]

    instance_txt_path = os.path.join(instance_txt_out_dir, image_dir + ".txt")
    instance_txt_file = open(instance_txt_path, 'a')


    masks = []
    track_ids = []

    for class_id in tracking_dict:
        converted_class_id = convert_class_id(class_id, class_names)
                
        if converted_class_id is None:
            continue

        for track_id in tracking_dict[class_id]:

            tracking_obj =  tracking_dict[class_id][track_id]

            if tracking_obj.is_found():
                mask = tracking_obj.get_mask()
                track_id = tracking_obj.get_id()

                converted_track_id = converted_class_id * id_divisor + track_id

                masks.append(mask)
                track_ids.append(converted_track_id)

    if len(masks) == 0:
        return



    if to_remove_overlap:
        masks = remove_overlap(masks)

    for track_id, mask in zip(track_ids, masks):
        mask = np.asfortranarray(mask)    
        mask = rletools.encode(mask)
        class_id = int(track_id / id_divisor)

        print(t, track_id, class_id, mask["size"][0], mask["size"][1],
              mask["counts"].decode(encoding='UTF-8'), file=instance_txt_file)

    instance_txt_file.close()

