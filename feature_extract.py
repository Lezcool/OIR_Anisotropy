import numpy as np
from kit.mics import *
import json
from tqdm.auto import tqdm

counts = 0
data_path = 'data/'
results_path = 'results/'
pic_dict = np.load(os.path.join(data_path,'materials_pic.npz'),allow_pickle=True)['materials_pic'].item()
isolabel = json.load(open(os.path.join(data_path,'isolabel.json')))

csv_folder = 'extracted_features/'
name_list = [x.split(".")[0] for x in os.listdir(os.path.join(data_path,csv_folder))]

new_material = []
for key in pic_dict.keys():
    if key not in name_list:
        new_material.append(key)

for n,key in enumerate(tqdm(new_material)):
    results_array = np.empty((0,14))  # new empty array for each new material
    counts = 0
    for k in tqdm(range(20),leave=False):   #data aug
        tf = 0 if k == 0 else 1
        for j,img in enumerate(pic_dict[key]):
            tmp = get_ae_2d(img,key,isolabel[key],j+counts,tf)
            results_array = np.vstack((results_array,tmp)) 
        counts += pic_dict[key].shape[0]
    all_aug_df = pd.DataFrame(results_array)
    all_aug_df.columns = ['Sample No',"Material","Layer","Intensity",'Intensity_index','Eccentricity','Ellipse angle','pca_eps','pca_angle','isotropy label','blur','transform','evr1','evr2']
    all_aug_df.to_csv((os.path.join(data_path,f'{csv_folder}{key}.csv')))