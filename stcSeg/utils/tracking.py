import os
import sys
import numpy as np
import math

sys.path.append('../mots_tools')
import pycocotools.mask as rletools

from detectron2.utils.colormap import random_color


def normalization(heatmap, target_min=-1, target_max=1):

    input_min = np.min(heatmap[np.nonzero(heatmap)])
    heatmap[np.nonzero(heatmap)] = heatmap[np.nonzero(heatmap)] - input_min

    input_max = np.max(heatmap)

    heatmap = heatmap / input_max * (target_max - target_min) + target_min

    return heatmap


def get_3D_position(point, depth_data, H_c=1.65, H_o=180, d=620, depth_factor=30, theta=0.5):

    W_p = depth_data.shape[1]
    
    theta = theta

    x = point[0]
    y = point[1]

    D = depth_factor / depth_data[x][y][0]
    H = H_c - D * (x - H_o + y * math.tan(math.pi/180 * theta)) / d
    Z = D * (y - W_p / 2) / d

    return D, H, Z



def extract_features(feature_model, image, boxes, resize=128, box_mode="XYXY_ABS"):
    import torch
    import torchvision.transforms as transforms
    import cv2

    img_to_tensor = transforms.ToTensor()

    # print("image:", image.shape)

    instance_img_list = []

    for box in boxes:
        if box_mode == "XYWH_ABS":
            x, y, w, h = box[0], box[1], box[2], box[3]
            x_1, y_1 = x + w, y + h
        elif box_mode == "XYXY_ABS":
            x, y, x_1, y_1 = box[0], box[1], box[2], box[3]
        instance_img = image[int(y):int(y_1),int(x):int(x_1),:]
        instance_img = cv2.cvtColor(np.asarray(instance_img), cv2.COLOR_RGB2BGR)
        instance_img = cv2.resize(instance_img, (resize, resize))
        instance_img = img_to_tensor(instance_img).unsqueeze(0)
        instance_img_list.append(instance_img)
    
    instance_img_list = torch.cat(instance_img_list, dim=0)
    features = feature_model(instance_img_list)

    return features.detach().numpy()


def get_feature_diff(feature_1, feature_2, method="euclidean"):
    from scipy.spatial import distance
    if method == 'euclidean':
         return distance.euclidean(feature_1, feature_2)
    elif method == 'cosine':
        return distance.cosine(feature_1, feature_2)
    elif method == 'dice':
        return distance.dice(feature_1, feature_2)
    else:
        raise "feature difference method error"




class TrackingInfo(object):
    def __init__(self, x, y, x1=None, y1=None, d=None, f=None, mask=None, color=None, track_id=None, score=None, survival=8):  # destroyed after 5 frame
        self.x = x
        self.y = y
        
        self.x1 = x1
        self.y1 = y1

        self.d = d
        self.f = f

        self.delta_x = 0
        self.delta_y = 0
        self.delta_x1 = 0
        self.delta_y1 = 0
        
        self.survival = survival
        self.found = True

        self.mask = mask
        self.color = color
        self.track_id = track_id
        self.score = score

    def update(self, x, y, x1=None, y1=None, d=None, f=None, mask=None, score=None, survival=8): 
        self.delta_x = x - self.x
        self.delta_y = y - self.y
        self.delta_x1 = x1 - self.x1
        self.delta_y1 = y1 - self.y1

        self.x = x
        self.y = y 

        self.x1 = x1
        self.y1 = y1

        self.d = d
        self.f = f

        self.survival = survival
        self.found = True

        self.mask = mask
        self.score = self.score

    def update_feature(self, feature):
        self.feature = feature

    def forward_a_step(self):
        self.survival -= 1
        self.found = False
        
        return self.survival

    def get_feature(self):
        return self.feature

    def get_data(self):
        if self.x1 is not None:
            return self.x + self.delta_x, self.y + self.delta_y, self.x1 + self.delta_x1, self.y1 + self.delta_y1, self.d, self.f
        else:
            return self.x + self.delta_x, self.y + self.delta_y
        

    def get_color(self):
        return self.color

    def get_mask(self):
        return self.mask

    def get_id(self):
        return self.track_id

    def get_score(self):
        return self.score

    def is_found(self):
        return self.found

    def is_survival(self):
        return self.survival >= 0



def track_objects(predictions, tracking_dict, depth_data=None, flow_data=None, feature_model=None, image=None, mask_tracking_distance=200, box_mode="XYXY_ABS", tracking_mode="DoublePoint+SD", survival=9):

    for class_id in tracking_dict:
        for track_id in tracking_dict[class_id]:
            tracking_dict[class_id][track_id].forward_a_step()

    masks = np.asarray(predictions.pred_masks)
    # print("masks:", masks.shape)
    classes = predictions.pred_classes
    # print(classes)
    boxes = predictions.pred_boxes

    # scores = predictions.scores.tolist()
    scores = predictions.scores

    scores = [float(score) for score in scores]

    if len(masks) == 0:
        return tracking_dict, [], [], []

    if depth_data is not None:
        # depth_data = normalization(depth_data, 100, 300)
        # print("depth:", depth_data.shape)
        assert (depth_data.shape[0] - masks.shape[1]) * (depth_data.shape[1] - masks.shape[2]) >= 0 

    if flow_data is not None:
        # flow_data = normalization(flow_data, 100, 300)
        # print("flow:", flow_data.shape)
        assert (flow_data.shape[0] - masks.shape[1]) * (flow_data.shape[1] - masks.shape[2]) >= 0 

    height = masks.shape[1]
    width = masks.shape[2]


    this_positions = []

    if feature_model is not None:
        assert image is not None
        features = extract_features(feature_model, image, boxes)

    for box in boxes:

        if box_mode == "XYWH_ABS":
            x, y, w, h = box[0], box[1], box[2], box[3]
            c_x = int(x + w / 2)
            c_y = int(y + h / 2)
            x_1, y_1 = x + w, y + h
        elif box_mode == "XYXY_ABS":
            x, y, x_1, y_1 = box[0], box[1], box[2], box[3]
            c_x = int((x + x_1) / 2)
            c_y = int((y + y_1) / 2)

        # print("cx: %f, cy: %f" % (c_x, c_y))

        if depth_data is not None:
            # d = depth_data[int(c_y / height * depth_data.shape[0])][int(c_x / width * depth_data.shape[1])] * 1
            d = depth_data[int(c_y / height * depth_data.shape[0])][int(c_x / width * depth_data.shape[1])] * 1.5
        else:
            d = 0

        if flow_data is not None:
            f = flow_data[int(c_y / height * flow_data.shape[0])][int(c_x / width * flow_data.shape[1])] * [0.01, 0.1]
            # print("flow data:", f)
        else:
            f = 0

        if tracking_mode == "CenterPoint":
            this_positions.append((c_x, c_y, 0, 0, 0, 0))
        elif tracking_mode == "DoublePoint":
            this_positions.append((x, y, x_1, y_1, 0, 0))
        elif tracking_mode == "DoublePoint+SD":
            this_positions.append((x, y, x_1, y_1, d, f))

    colors = 'bgrcmyk'

    color_bgr = {'b': [255, 0, 0], 'g': [0, 255, 0], 'r': [0, 0, 255], 
                 'c': [255, 255, 10], 'm': [255, 0, 255], 'y': [0, 255, 255],
                 'k': [0, 0, 0]}

    matching_maps = {}

    # found the min d for objects in obj_list
    for class_id in tracking_dict:

        matching_maps[class_id] = {}

        # found the min d for objects in tracking_dict
        for track_id in tracking_dict[class_id]:
            previous_obj = tracking_dict[class_id][track_id]

            if not previous_obj.is_survival():
                continue

            pr_x, pr_y, pr_x1, pr_y1, pr_d, pr_f = previous_obj.get_data()
            # print("Previous: ", previous_obj.get_data())

            min_distance = max(abs(pr_x1 - pr_x), abs(pr_y1 - pr_y)) * 2 # mask_tracking_distance  # float("inf")
            matching_id = None

            for this_id, (x, y, x1, y1, d, f) in enumerate(this_positions):
                # print("This: ", this_positions[this_id])
                this_class_id = classes[this_id]
                if this_class_id == class_id:
                    # To Do: more functions!!!!!!!!!!!!
                    distance = ((pr_x - x) ** 2 + (pr_y - y) ** 2 + (pr_x1 - x1) ** 2 + (pr_y1 - y1) ** 2 ) ** 0.5
                    # distance = (pr_x - x) ** 2 + (pr_y - y) ** 2 + (pr_x1 - x1) ** 2 + (pr_y1 - y1) ** 2 
                    if depth_data is not None:
                        distance = distance + abs(pr_d - d)
                        # distance = distance + (pr_d - d) ** 2
                    if flow_data is not None:
                        distance = distance + ((pr_f[0] - f[0]) ** 2 + (pr_f[1] - f[1]) ** 2) ** 0.5
                        # distance = distance + (pr_f[0] - f[0]) ** 2 + (pr_f[1] - f[1]) ** 2

                    if feature_model is not None:
                        pr_feature = previous_obj.get_feature()
                        distance = distance + get_feature_diff(features[this_id], pr_feature)

                    if distance < min_distance:
                        min_distance = distance
                        matching_id = this_id

            matching_maps[class_id][track_id] = matching_id
            # print("%d -> %d" % (track_id, matching_id))

    track_id_list = []
    assigned_colors = []
    info_list = []

    not_found_ids = []

    for this_id, (x, y, x1, y1, d, f) in enumerate(this_positions):

        class_id = int(classes[this_id])

        if class_id not in tracking_dict.keys():
            tracking_dict[class_id] = {}


        found_id = None
        min_distance = max(abs(x1 - x), abs(y1 - y)) * 2 # mask_tracking_distance  # float("inf")

        if class_id in matching_maps.keys():
            for track_id in matching_maps[class_id]:
                if matching_maps[class_id][track_id] == this_id:
                    previous_obj = tracking_dict[class_id][track_id]
                    
                    if previous_obj.is_found() or not previous_obj.is_survival():
                        continue
                    
                    pr_x, pr_y, pr_x1, pr_y1, pr_d, pr_f = previous_obj.get_data()
                    distance = ((pr_x - x) ** 2 + (pr_y - y) ** 2 + (pr_x1 - x1) ** 2 + (pr_y1 - y1) ** 2 ) ** 0.5
                    # distance = (pr_x - x) ** 2 + (pr_y - y) ** 2 + (pr_x1 - x1) ** 2 + (pr_y1 - y1) ** 2 
                    if depth_data is not None:
                        distance = distance + abs(pr_d - d)
                        # distance = distance + (pr_d - d) ** 2 
                    if flow_data is not None:
                        distance = distance + ((pr_f[0] - f[0]) ** 2 + (pr_f[1] - f[1]) ** 2) ** 0.5
                        # distance = distance + (pr_f[0] - f[0]) ** 2 + (pr_f[1] - f[1]) ** 2

                    if feature_model is not None:
                        pr_feature = previous_obj.get_feature()
                        distance = distance + get_feature_diff(features[this_id], pr_feature)
                        # print("distance:", distance)
                        # print("feature diff:", get_feature_diff(features[this_id], pr_feature))

                    if distance < min_distance:
                        min_distance = distance
                        found_id = track_id

        if found_id is not None:
            # print("%d -> %d" % (this_id, found_id))
            tracking_dict[class_id][found_id].update(x, y, x1=x1, y1=y1, d=d, f=f, mask=masks[this_id], score=scores[this_id], survival=survival)
            if feature_model is not None:
                tracking_dict[class_id][found_id].update_feature(features[this_id])
            
            # print(found_id)
            color = tracking_dict[class_id][found_id].get_color()

        else:
            # print("%d -> []" % (this_id,))
            not_found_ids.append(this_id)
            color = None

        track_id_list.append(found_id)
        assigned_colors.append(color)
        # info_list.append("%.1f %.1f %.1f" % (x, y, z))
        info_list.append("")

    for this_id in not_found_ids:

        class_id = int(classes[this_id])
        x, y, x1, y1, d, f = this_positions[this_id]

        min_distance = max(abs(x1 - x), abs(y1 - y)) * 2 # mask_tracking_distance  # float("inf")
        found_id = None

        for track_id in tracking_dict[class_id].keys():
            previous_obj = tracking_dict[class_id][track_id]

            if not previous_obj.is_survival() or previous_obj.is_found():
                continue

            pr_x, pr_y, pr_x1, pr_y1, pr_d, pr_f = previous_obj.get_data()
            
            # To Do: more functions!!!!!!!!!!!!
            distance = ((pr_x - x) ** 2 + (pr_y - y) ** 2 + (pr_x1 - x1) ** 2 + (pr_y1 - y1) ** 2 ) ** 0.5
            # distance = (pr_x - x) ** 2 + (pr_y - y) ** 2 + (pr_x1 - x1) ** 2 + (pr_y1 - y1) ** 2 
            if depth_data is not None:
                distance = distance + abs(pr_d - d)
                # distance = distance + (pr_d - d) ** 2
            if flow_data is not None:
                distance = distance + ((pr_f[0] - f[0]) ** 2 + (pr_f[1] - f[1]) ** 2) ** 0.5
                # distance = distance + (pr_f[0] - f[0]) ** 2 + (pr_f[1] - f[1]) ** 2

            if feature_model is not None:
                pr_feature = previous_obj.get_feature()
                distance = distance + get_feature_diff(features[this_id], pr_feature)

            if distance < min_distance:
                min_distance = distance
                found_id = track_id

        if found_id is not None:
            tracking_dict[class_id][found_id].update(x, y, x1=x1, y1=y1, d=d, f=f, mask=masks[this_id], score=scores[this_id], survival=survival)
            if feature_model is not None:
                tracking_dict[class_id][found_id].update_feature(features[this_id])

        else:

            if len(tracking_dict[class_id].keys()) != 0:
                new_track_id = max(tracking_dict[class_id].keys()) + 1
            else:
                new_track_id = 0

            # color = color_bgr[colors[(new_track_id) % len(colors)]]
            color = random_color(rgb=True, maximum=1)

            tracking_dict[class_id][new_track_id] = TrackingInfo(x, y, x1=x1, y1=y1, d=d, f=f, mask=masks[this_id], color=color, track_id=new_track_id, score=scores[this_id], survival=survival)
            if feature_model is not None:
                tracking_dict[class_id][new_track_id].update_feature(features[this_id])
            found_id = new_track_id

        # print(found_id)
        color = tracking_dict[class_id][found_id].get_color()

        track_id_list[this_id] = found_id
        assigned_colors[this_id] = color
        # info_list[this_id] = "%.1f %.1f %.1f" % (x, y, z)
        info_list[this_id] = ""


    return tracking_dict, track_id_list, assigned_colors, info_list


def convert_class_id(class_id, class_names, dataset="KITTI_MOTS"):

    assert class_names is not None

    if class_id > len(class_names):
        # print("Class id error: ", class_id)
        return None

    # print("class_id: %d" % class_id)

    if dataset == "KITTI_MOTS":
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

    # print("class_id: %d" % class_id)

    return class_id


def remove_overlap(masks, scores):
    assert len(masks) == len(scores)

    if len(masks) == 0:
        return []
    
    h = masks[0].shape[0]
    w = masks[0].shape[1]

    # to solve overlapped masks

    # scored_all_masked = np.zeros((h, w))
    # new_masks = []

    # for old_mask, score in zip(masks, scores):

    #     scored_old_mask = old_mask * score
        
    #     scored_mask = np.maximum(scored_old_mask - scored_all_masked, 0)
    #     scored_mask[scored_mask > 0] = score

    #     scored_all_masked = np.maximum(scored_all_masked - scored_mask, 0)
    #     scored_all_masked += scored_mask

    #     scored_mask[scored_mask > 0] = 1

    #     mask = scored_mask.astype(np.uint8)
    #     mask = np.asfortranarray(mask)
    #     new_masks.append(mask)

    # masks = new_masks[::-1]

    new_masks = []
    all_masked = np.zeros((h, w))

    for old_mask in masks:
        
        mask = old_mask - all_masked
        mask = np.maximum(mask, 0)
        all_masked += mask

        mask = mask.astype(np.uint8)
        mask = np.asfortranarray(mask)
        new_masks.append(mask)

    # return new_masks[::-1]
    return new_masks



def save_instance(tracking_dict, path, class_names, dataset, instance_txt_out_dir=None, to_remove_overlap=True, id_divisor=1000):

    # print(class_names)

    img_name = path.split('/')[-1].split('.')[0]
    t = int(img_name)

    image_dir = path.split('/')[-2]

    if instance_txt_out_dir is not None and instance_txt_out_dir != "":
        instance_txt_path = os.path.join(instance_txt_out_dir, image_dir + ".txt")
        instance_txt_file = open(instance_txt_path, 'a')


    masks = []
    track_ids = []
    scores = []

    for class_id in tracking_dict:
        converted_class_id = convert_class_id(class_id, class_names, dataset)
                
        if converted_class_id is None:
            continue

        for track_id in tracking_dict[class_id]:

            tracking_obj =  tracking_dict[class_id][track_id]

            if tracking_obj.is_found():
                mask = tracking_obj.get_mask()
                track_id = tracking_obj.get_id()
                score = tracking_obj.get_score()

                # if score < score_threshold:
                #     continue

                converted_track_id = converted_class_id * id_divisor + track_id

                masks.append(mask)
                track_ids.append(converted_track_id)
                scores.append(score)

    if len(masks) == 0:
        return



    if to_remove_overlap:
        masks = remove_overlap(masks, scores)

    for track_id, mask, score in zip(track_ids, masks, scores):
        mask = np.asfortranarray(mask)    
        mask = rletools.encode(mask)
        class_id = int(track_id / id_divisor)

        if instance_txt_out_dir is not None and instance_txt_out_dir != "":
            print(t, track_id, class_id, mask["size"][0], mask["size"][1],
                mask["counts"].decode(encoding='UTF-8'), score, file=instance_txt_file)

    if instance_txt_out_dir is not None and instance_txt_out_dir != "":
        instance_txt_file.close()
