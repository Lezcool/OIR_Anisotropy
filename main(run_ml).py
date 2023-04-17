import os
import warnings
warnings.filterwarnings("ignore")


from kit.mics import *
from kit.Machine_learning import *

data_path = 'data/'
results_path = 'results/'


params = 'default'
merge_list = 'no'
names_classifiers = 'default'
selected_material = 'all'

selected_df = read_material_csv(data_path,merge_list,selected_material)

#write setting to txt file
save_folder = 'results/'
if not os.path.exists(os.path.join(results_path,save_folder)):
    os.makedirs(os.path.join(results_path,save_folder))
    print('Make new folder due to directory not exist')

with open(os.path.join(results_path,save_folder,'setting.txt'),'w') as f:
    for data,name in zip([names_classifiers,selected_material,merge_list,params],['names_classifiers','selected_material','merge_list','params']):
        f.write(name)
        f.write('\n')
        f.writelines(str(data))
        f.write('\n,\n')
        print(name,'\n',data,'\n','*'*30)
    f.write(results_path)

train_mymodel(selected_df, results_path,
    names_classifiers=names_classifiers,
    epochs=50,params=params,extra_cmt='default',fixseed=True)
