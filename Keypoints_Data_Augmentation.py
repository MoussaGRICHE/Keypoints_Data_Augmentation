import albumentations as A
import cv2
import json
import os
import Paths

Images_names = os.listdir(Paths.Images_Path) # a list of all images names

##################################################################################################################
# Read json file
##################################################################################################################

json_data = open(Paths.Json_Path)
json_data = json.load(json_data)

##################################################################################################################
# Read all images
##################################################################################################################

Image_dataset = {}
for img in Images_names:
    Image_dataset[str(img)] = cv2.imread(Paths.Images_Path+"/"+ str(img))
    Image_dataset[str(img)] = cv2.cvtColor(Image_dataset[str(img)], cv2.COLOR_BGR2RGB)

##################################################################################################################
# Create a list of dictionnaires with the importantes informations. Each dictionnairy represents one annotation informations.
##################################################################################################################

image_size = json_data["images"]
annotations_size = json_data["annotations"]
kps=[]
for i in range(len(annotations_size)):
    dic = dict()
    for j in range(len(image_size)):
        dic["image_id"] = json_data["annotations"][i]["image_id"]
        dic["annotation_id"] = json_data["annotations"][i]["id"]
        dic["category_id"] = json_data["annotations"][i]["category_id"]
        dic["keypoints"] = []
        L = json_data["annotations"][i]["keypoints"]
        for p in range(0, len(L) - 2, 3):
            res = (L[p], L[p + 1], L[p + 2])
            dic["keypoints"].append(res)
        if json_data["annotations"][i]["image_id"] == json_data["images"][j]["id"]:
            dic["image_name"] = json_data["images"][j]["file_name"]
    kps.append(dic)

##################################################################################################################
# Transformation models
##################################################################################################################

transform_V = A.Compose([A.VerticalFlip(p=1), A.RandomBrightnessContrast(p=0.2)], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
transform_H = A.Compose([A.HorizontalFlip(p=1), A.RandomBrightnessContrast(p=0.2)], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

##################################################################################################################
#Transformation images and keyspoints
##################################################################################################################

model = input("What transformation model you want. Put VT for vertical transformation and HT for horizontal transformation: ")

New_KP = []
New_image_dataset = {}

for j in Images_names:
    for i in range(len(kps)):
        Kp_image = dict()
        img = Image_dataset[j]
        if kps[i]["image_name"] == j:
            Kp_image['image_name']= model+str(j)
            Kp_image['image_id'] = kps[i]['image_id']
            Kp_image['annotation_id'] = kps[i]['annotation_id']
            Kp_image['category_id'] = kps[i]['category_id']
            keypoints = kps[i]["keypoints"]
            if model == str("VT"):
                transformed = transform_V(image=img, keypoints=keypoints)
                transformed_image = transformed['image']
                transformed_keypoints = transformed['keypoints']
                out = [item for t in transformed_keypoints for item in t]  # transforme a list of tuple to a list
                for p in range(0, len(out) - 2,
                               3):  # if the keypoint is out of the image (0,0,0), the new keypoint stille out of the image (0,0,0)
                    if out[p] == 0 and out[p + 2] == 0:
                        out[p + 1] = 0
            if model == str("HT"):
                transformed = transform_H(image=img, keypoints=keypoints)
                transformed_image = transformed['image']
                transformed_keypoints = transformed['keypoints']
                out = [item for t in transformed_keypoints for item in t] # transforme a list of tuple to a list
                for p in range(0, len(out) - 2, 3): # if the keypoint is out of the image (0,0,0), the new keypoint stille out of the image (0,0,0)
                    if out[p+1] == 0 and out[p + 2] == 0:
                        out[p] = 0
            Kp_image['keypoints'] = out # the list of the news keypoints
            cv2.imwrite(os.path.join(Paths.New_Images_Path , model+str(j)), transformed_image) # save the new image in a new directory
            New_KP.append(Kp_image)

    New_image_dataset[model+str(j)] = transformed_image

##################################################################################################################
# Saving new keypoints in a new json file
##################################################################################################################

with open(Paths.New_Json_Path, 'r+') as f:
    data = json.load(f)
    for i in range(len(data["images"])):
        data["images"][i]["file_name"] = model + str(json_data["images"][i]["file_name"])
        f.seek(0)
        json.dump(data, f, indent=4)
        f.truncate()
    for i in range(len(data["annotations"])):
        data["annotations"][i]["keypoints"] = New_KP[i]["keypoints"]
        f.seek(0)
        json.dump(data, f, indent=4)
        f.truncate()

##################################################################################################################

