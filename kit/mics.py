import pandas as pd
import os
import random
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.ndimage import gaussian_filter
import torchvision.transforms.functional as TF
from sklearn.decomposition import PCA
import cv2
import torchvision.transforms as T

def read_material_csv(csv_folder,merge=None,pick=None):
  '''
  csv files in data_path/material_csv/name.csv
  merge = [('a','b'),('c','d')] or merege == 'no'
  pick = ['a','b','c','d'] or pick == 'all'
  '''
  name_list = [x.split(".")[0] for x in os.listdir(csv_folder)]
  df = pd.DataFrame()
  if pick =='all' and merge =='no':
    for name in name_list:
      tmp_df = pd.read_csv(os.path.join(csv_folder,f'{name}.csv')).drop('Unnamed: 0',axis=1)
      df = pd.concat([df,tmp_df],ignore_index=True)
    return df
  print('*'*20)
  print('All materials below:')
  print(sorted(name_list))
  print('*'*20)
  if merge ==None or pick ==None: 
    assert print('input merge list or pick list')


  def merge_material(merge_l1):
    df = pd.DataFrame()
    add_num = 0
    for idx,name in enumerate(merge_l1):
      tmp_df = pd.read_csv(f'{csv_folder}{name}.csv').drop('Unnamed: 0',axis=1)
      tmp_df['Sample No'] += add_num
      add_num += tmp_df['Sample No'].max() + 1
      tmp_df['Material'] = merge_l1[0]
      df = pd.concat([df,tmp_df],ignore_index=True)
    return df
  
  if merge == 'no':
    merge_df = pd.DataFrame()
  else:
    merge_df = pd.DataFrame()
    for merge_l1 in merge:
      if isinstance(merge_l1,list):
        tmp_df = merge_material(merge_l1)
        merge_df = pd.concat([merge_df,tmp_df],ignore_index=True)
      else:
        merge_df = merge_material(merge)
        break
      
  for p_name in pick:
    tmp_df = pd.read_csv(f'{csv_folder}{p_name}.csv').drop('Unnamed: 0',axis=1)
    df = pd.concat([df,tmp_df],ignore_index=True)

  df = pd.concat([df,merge_df],ignore_index=True)
  print('Materials in the df:')
  print(df['Material'].unique())
  print('*'*20)
  return df


def seperate_pca_ellipse_results(df):
    '''
    input has old name condition
    return df with a new column methods to seperate pca and ellipse
    '''
    df.loc[df.condition.str.contains('elps'),'methods'] = 'ellipse'
    df.loc[df.condition.str.contains('pca'),'methods'] = 'pca'
    for rps in ['pca_','elps_']:
        df.loc[df.condition.str.contains(rps),'condition'] = df.condition.str.replace(rps,'')

    return df


def randsamp(s:int,e:int,n:int,seed:int = None):
  """
  s:start, e:end, n: pick n samples
  retrun picked, rest
  """
  if seed:
    random.seed(seed)
  rest = [i for i in range(s,e)]
  picked = sorted(random.sample(range(s,e),n))
  for i in picked:
    rest.remove(i)
  return picked,rest


def save_fig(df,save_path,x=None,y=None,filename=None):
    t = time.strftime("%Y%m%d-%H%M", time.localtime()) 
    if x ==None and y==None:
        print(df.columns)
        x = input('select x: ')
        y = input('select y: ')
    plt.figure(figsize=(15,5))
    try:
        sns.lineplot(data = df,x =x, y=y,hue='condition',style='methods')
    except:
        sns.lineplot(data = df,x =x, y=y,hue='condition')
    plt.xticks(rotation=-35)
    plt.title(y)
    if filename==None:
        plt.savefig(os.path.join(save_path,'%s_%s.pdf'%(y,t)),bbox_inches='tight')
    else:
        plt.savefig(os.path.join(save_path,'%s_%s_%s.pdf'%(filename,y,t)),bbox_inches='tight')

def mytransform(img4c,mode='color'):
  '''
  img has 4 chennel, (4,400,400)
  '''
  img4c = [TF.to_pil_image(-TF.to_tensor(x)) for x in img4c]
  #for i in img4c: i=TF.to_pil_image(-TF.to_tensor(i))
  
  #Random rotate
  rotater = T.RandomRotation(degrees=(0, 360))
  img4c = [rotater(i) for i in img4c]
  
  if mode == 'color':  #adjust color
     jitter = T.ColorJitter(brightness=[0.5,1.5], contrast=[0.5,1.5],saturation=[0.5,1.5])
     img4c = [jitter(i) for i in img4c]

  # perspective_transformer = T.RandomPerspective(distortion_scale=0.2, p=0.3)
  # img4c = [perspective_transformer(i) for i in img4c]
  #pil to tensor
  img4c = [np.squeeze(TF.pil_to_tensor(x).numpy()) for x in img4c] 
  
  return img4c

def get_intensity_range(img):
  '''
  img (400,400)
  return [start, end, step]
  '''
  start,end = None, None
  minp,maxp = int(np.min(img)),int(np.max(img))
  for intensity in range(minp,maxp,5):
    num_points = len(np.where(img>=intensity)[0])
    if num_points <= 120000:
      if start == None:
        start = intensity
    if num_points <= 10000:
      if end == None:
        end = intensity
  if start ==None: start = minp
  if end == None or end >240: end = maxp
  step = (end-start)//10

  return start,start+step*10+1,step

def roibox(img,l = 400):
    if img.shape[1] < 500: return img
    tmp = np.where(img>=np.max(img)-10)  #也可以试试看peak_local_max?
    x_var,y_var = np.var(tmp[0]),np.var(tmp[1])
    if x_var>5000 or y_var > 5000:
        img_resize = roibox(img[1000:2500,2000:4000])
        plt.title('Auto selected roibox')
        plt.imshow(img_resize)
        plt.show()
        loop = input('if it is correct, input t else f: ')
        while loop != 't':
            print('possible more than 1 cluster, please input x,y limit')
            plt.title('Original image')
            plt.scatter(tmp[1],tmp[0])
            plt.imshow(img)
            plt.show()
            try:
                a, b = [int(x) for x in input('x: a,b: ').split(',')]
                c, d = [int(x) for x in input('y: c,d: ').split(',')]
                img_resize = roibox(img[c:d,a:b])  #img x is vertical
                plt.imshow(img_resize)
                plt.show()
                loop = input('if it is correct, input t else f: ')
            except:
                loop == 'f'
            
        return img_resize

    y_mid = int(np.mean(tmp[0]))
    x_mid = int(np.mean(tmp[1]))
    hl = int(l/2)
    img_resize = img[y_mid-hl: y_mid+hl, x_mid-hl: x_mid+hl]
    #plt.imshow(img_resize,cmap='gray')
    return img_resize

def get_ae_2d(img, material, ai, sample_no , tf=0, blur=0):
  '''#input is raw image. (4,400,400)
  blur =1, apply, blur=0 nothing
  ai: isotropy=0, anisotropy=1
  tf: tranform = 1 apply tranformation
  #minrange=20,maxrange=221,step=10 means the pixel intensity
  return list ['Sample No',"Material","Band","Intensity",'Eccentricity','Ellipse angle','pca_eps','pca_angle','isotropy label','blur','transform']
  '''
  #fig,axes =plt.subplots(4,4,figsize=(20,20))
  light = ['band0 violet','band1 violet','band2 red','band3 red']
  results = []
  num = 0

  if tf == 1:
    step_check = [0]
    while 0 in step_check:
      step_check = []
      img_tf = mytransform(img)
      for im in img:
        _,_,step_tmp = get_intensity_range(im)
        step_check.append(step_tmp)
  elif tf == 0:
    img_tf = img

  for im in img_tf:
    im = roibox(im)
    if blur==1:
      im = gaussian_filter(im.reshape(400,400,1),[3,3,0]).reshape(400,400)
    num += 1   #num is band
    start,end,step = get_intensity_range(im)
    if step == 0:
      print(material)
      plt.imshow(im)
      plt.show()
      print(start,end,step)
    
    for intensity_idx,intensity in enumerate(range(start,end,step)):
      imgcopy = im.copy()
      points = list(np.where(imgcopy>=intensity)) #(2,n)

      points[0],points[1] = points[1], points[0] #img(H,W) to (W,H)
      points = np.array(points)  #(2.n)
      if points.shape[1] < 10:
        plt.imshow(imgcopy)
        print(start,end,step,intensity,points)
        return imgcopy
      #print(points.shape)
      el_params = cv2.fitEllipse(points.T)
      #for y, downside is +
      #x,y is inversed
      x,y = el_params[0] #ellipse center
      b,a = el_params[1] # a,b is fixed
      angle = el_params[2]
      eps = (1-(b/a)**2)**0.5 #eccentricity

      #PCA
      x1,y1 = points.mean(axis=1)
      pca = PCA(n_components=2).fit(points.T) 
      comp, sv, sv1 = pca.components_[0], pca.singular_values_[0],pca.singular_values_[1]
      evr = pca.explained_variance_ratio_
      pradians = np.arctan(comp[1]/comp[0])
      sv, sv1 = max(sv,sv1), min(sv,sv1)
      pca_eps = (1-(sv1/sv)**2)**0.5
      pca_angle = np.degrees(pradians)
      if pca_angle < 90: pca_angle += 90

      
      results.append([sample_no,material,num,intensity,intensity_idx,eps,angle,pca_eps,pca_angle,ai,blur,tf,evr[0],evr[1]]) #band,intensity,eps,angle

  return np.array(results)