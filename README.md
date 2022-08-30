# Keypoints_Data_Augmentation
Keypoints coco dataset augmentation with Albumentations
This Script does the data augmentation using two models from Albumentations library.

The two models are: Vertical transformations and Horizontal transformations.

You should put the path for the images folder and the path for the original json file.
Then, you must create a new folder for the news images and put his path. 
In add of this, you have to create a copy of the original json file and put his path 

Put all paths in new python file named "Paths.py" as:
* Images_Path = ""
* Json_Path = ""
* New_Images_Path = ""
* New_Json_Path = ""

If you want to select "Vertical transformation", you should write "VT" when the input message displaye on the terminal. 
Also, If you want to select "Vertical transformation", you should write "HT" when the input message displaye on the terminal.