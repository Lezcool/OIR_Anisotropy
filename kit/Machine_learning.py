from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Model
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
import random
import os
import sys
from tqdm.auto import tqdm



# evaluate
from sklearn.metrics import roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
import pandas as pd
from sklearn.decomposition import PCA

from kit.mics import *



def get_y(data_df,index=['Sample No'],y_label =['Material','isotropy label']):
    y = data_df.groupby(index).agg(lambda x: list(x)[0])[y_label]
    return y


def apply_pca(data,values=['Ellipse angle','Eccentricity'],index=['Sample No'],columns=['Intensity','Band'],y_label =['Material','isotropy label'],new_D=0):
    '''
    return [idx,x,y] pd dataframe
    '''
    pivot = pd.pivot_table(data, values=values,index=index,columns=columns)
    y = data.groupby(index).agg(lambda x: list(x)[0])[y_label].reset_index() #get material name and iso label

    n_features = len(values)
    n_samples = len((pivot.iloc[0]).values)//n_features
    if new_D == 0:
        new_D = min(n_samples, n_features)
    es = PCA(n_components=new_D)
    new_array = np.empty((0,new_D))

    for r in range(pivot.shape[0]):
        tmp = np.array(pivot.iloc[r]).reshape(n_features,-1).T
        es.fit(tmp)
        result = es.singular_values_
        new_array = np.vstack((new_array,result))
    ea = pd.DataFrame(new_array)
    if new_D == n_features:
        ea.columns = values
    pca_pd = ea.join(y)

    return pca_pd

       
def clr_compare(train_xy,label, test_xy=None,names_classifiers='default',norm=False,miscls=False,all_material=None):
    '''
    names_classifiers='default' or [(name1,clf1),(name2,clf2)]
    train_y  (n_sample,1): iso label, (n,2): material, iso label 
    test_y set to -1 if do not know
    '''

    # for miss material calculate

    train_x = train_xy.drop(columns=[label,'Material'])
    train_y = train_xy[label]
    if type(test_xy) != type(None):
        test_x = test_xy.drop(columns=[label,'Material'])
        test_y = test_xy[label]
        if test_y.sum() < 0: test_y = None
    #make sure when graph is true, train has 2d shape
    if len(train_x.shape)==2 and train_x.shape[1] != 2: 
        if graph == True: print('train size is not 2D, graph set to False')
        graph=False

    #get misclassified material
    try:
        train_y_material = train_xy['Material']
        if type(test_y) != type(None):
            test_y_material = test_xy['Material']
    except:
        if miscls == True: print('y shape does not fit, miscls set to false')
        miscls=False

    #when multiclass y
    if train_y.value_counts().shape[0]>2:
        multicalss = True
        scoring_way = 'roc_auc_ovo'
    elif train_y.value_counts().shape[0]==2:
        multicalss = False
        scoring_way = 'roc_auc'
    else:
        assert print('y class does not fit')

    if train_y.shape[-1] ==1: # (n,) to ravel
        train_y=np.ravel(train_y)
        test_y=np.ravel(test_y)

    if type(test_x) != type(None) and type(test_y) == type(None): 
        pred_mode = True
        pred_results = {}
    else:
        pred_mode = False

    if names_classifiers=='default':
        names_classifiers = [
            ("Nearest Neighbors", KNeighborsClassifier()),
            ("Linear SVM", SVC(kernel="sigmoid", C=1)),
            ("RBF SVM",SVC(gamma='scale', C=1)),
            ("Logistic Regression",LogisticRegression(max_iter=5000,solver='newton-cg')),
            ("Decision Tree",DecisionTreeClassifier()),
            ('ExtraTrees',ExtraTreesClassifier()),
            ("Random Forest",RandomForestClassifier()),
            ("Neural Net",MLPClassifier()),
            ("Neural Net++",MLPClassifier(hidden_layer_sizes=(400,200,100,50,25),alpha=0.01, max_iter=10000,learning_rate='adaptive')),
            ("Neural Net++ES",MLPClassifier(hidden_layer_sizes=(400,200,100,50,25),alpha=1, max_iter=10000,learning_rate='adaptive',early_stopping=True,validation_fraction=0.3)),
            ("AdaBoost",AdaBoostClassifier(n_estimators=400)),
            ("Naive Bayes",GaussianNB()),
            ("QDA",QuadraticDiscriminantAnalysis()),
        ]

    if norm: train_x = StandardScaler().fit_transform(train_x)
    results = {}
    miss = []

    # iterate over classifiers
    for name, clf in names_classifiers:
        X_train, X_val, y_train, y_val = train_test_split(train_x, train_y, test_size=0.2)
        clf.fit(X_train, y_train)
        pred_y_val = clf.predict(X_val)
        val_score = roc_auc_score(y_val,pred_y_val)

        if type(test_x) != type(None) and type(test_y) != type(None):
            miss_counts = [name]
            pred_y = clf.predict(test_x)
            if miscls: 
                for material_name in all_material:
                    if list(test_y_material).count(material_name) != 0:
                        miss_counts.append(list(test_y_material[pred_y != test_y]).count(material_name)/list(test_y_material).count(material_name))
                    else:
                        miss_counts.append(0)
                miss.append(miss_counts)
            
            score = roc_auc_score(test_y,pred_y) if multicalss==False else roc_auc_score(test_y,pred_y,average='macro')
            results[name] = [round(x,4) for x in (val_score,score)]

        
        #predict mode
        elif type(test_x) != type(None) and type(test_y) == type(None): 
            print('Predict mode under construction')
            sys.exit()
            results[name] = round(cv.mean())
            pred_y = clf.predict(test_x)
            pred_results[name] = pred_y
        else:
            print('What mode is it??')
            sys.exit()
            results[name] = round(cv.mean())
            
    if miscls:
        miss_pd = pd.DataFrame(miss)
        miss_pd.columns = ['clr_name']+ all_material
        return results, miss_pd
    if pred_mode: return results, pred_results
    return results

def train_mymodel(df, save_path,params='default',epochs = 50,extra_cmt='default',names_classifiers='default',fixseed=True):
    all_material = list(df['Material'].unique())
    label = 'isotropy label'
    if len(df[label].unique()) >2:
        print('There are unreliable label:')
        print(df[label].unique())
        sys.exit()
    if params == 'default':
        index_list = [['Material','Sample No'],['Material','Sample No'],['Material','Sample No'],['Material','Sample No','Band']]
        value_list = [['Eccentricity','Ellipse angle'],['Ellipse angle'],['Eccentricity'],['pca_eps','pca_angle'],['pca_eps'],['pca_angle']]
        desc_ilist = ('4Band','2Bandv','2Bandr','1Band')
        desc_vlist = ('elps_ecc,angle','elps_angle','elps_ecc','pca_ecc,angle','pca_ecc','pca_angle')
    else:
        try:
            index_list,value_list,desc_ilist,desc_vlist = params
            if len(index_list) != len(desc_ilist) or len(value_list) != len(desc_vlist):
                return print('Error: params length unmatch')
        except:
            return print('Error: Wrong params')
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print('Make new folder due to directory not exist')

    
    leaderboard_df = pd.DataFrame()
    miss_all_df = pd.DataFrame()
    
    if fixseed:
        randlist = [391154, 762492, 481312, 383436, 203756, 863150, 658559, 383600,
       407817, 453821,  87165, 109580,  53129, 976148, 988969,  26262,
       652471, 447379, 542034, 332553, 404029, 346567, 339566, 542862,
       265880, 810137, 667418, 375901, 698650, 964398,  27846, 261242,
       337796, 599316, 742209, 110872, 913425, 234020, 959107, 147643,
       475481,  39153, 206433, 842765, 329998, 135753, 876920, 884438,
       779951, 168196]

    test_material_hist = []
    for iter in tqdm(range(epochs),position=0,file=sys.stdout):

        if fixseed:
            randseed = randlist[iter]
        else:
            randseed = random.randint(0,1000000)
        train_material, test_material = quick_sampling(df,ratio=0.4,seed=randseed)

        while sorted(test_material) in test_material_hist:
            randseed = random.randint(0,1000000)
            print(randseed)
            train_material, test_material = quick_sampling(df,ratio=0.4,seed=randseed)
            print(test_material)
        
        
        test_material_hist.append(sorted(test_material))
        
        train_m_df = df.query('Material == %s'%train_material)
        test_m_df = df.query('Material == %s and transform==0'%test_material)
        
        for j,index in enumerate(tqdm(index_list,desc='index list',leave=False)):
            if index == ['Material','Sample No']:
                column = ['Intensity_index','Band']
            elif index == ['Material','Sample No','Band']:
                column = ['Intensity_index']
            else:
                print('Error Wrong index')
                sys.exit()

            if desc_ilist[j] == '2Bandv':
                train_df, test_df = [df.query('Band == 1 or Band == 2') for df in [train_m_df, test_m_df]]
            elif desc_ilist[j] == '2Bandr' :
                train_df, test_df = [df.query('Band == 3 or Band == 4') for df in [train_m_df, test_m_df]]
            else:
                train_df, test_df = train_m_df, test_m_df

            for k,v in enumerate(tqdm(value_list,desc='value list',leave=False)):

                train_x,test_x = [pd.pivot_table(data=df, values=v,index=index,columns=column) for df in [train_df,test_df]]
                train_y, test_y = [get_y(data_df=df,index=index,y_label=[label]).reset_index()[label] for df in [train_df,test_df]]
                def mergexy(x,y):
                    x.columns = mergelevel(x)
                    xy = x.reset_index().join(y)
                    if 'Band' in list(xy.columns):
                        xy = xy.drop(columns=['Sample No','Band']).sample(frac=1)
                    else:
                        xy = xy.drop(columns=['Sample No']).sample(frac=1)
                    return xy
                train_xy,test_xy = [mergexy(x,y) for x,y in [(train_x,train_y),(test_x,test_y)]]

                results,miss_df = clr_compare(train_xy,label,test_xy,names_classifiers=names_classifiers,miscls=True,all_material=all_material)
                leaderboard = clr_result_to_DF(results)
                leaderboard['condition'] = desc_ilist[j]+','+desc_vlist[k]
                leaderboard['randomseed'] = randseed
                leaderboard['iter'] = iter

                leaderboard_df = pd.concat([leaderboard_df,leaderboard],ignore_index=True)
                miss_df['condition'] = desc_ilist[j]+','+desc_vlist[k]
                miss_df['iter'] = iter
                miss_all_df = pd.concat([miss_all_df, miss_df], ignore_index=True)
    
        if (iter+1)%5 == 0 or iter == 0 or iter == epochs:
            try:
                pd.DataFrame(test_material_hist).to_csv(os.path.join(save_path,'test_material.csv'))

                leaderboard_df_sp = seperate_pca_ellipse_results(leaderboard_df)
                leaderboard_df.to_csv(os.path.join(save_path,'mymodel_%s.csv'%(extra_cmt)))
                leaderboard_df = pd.read_csv(os.path.join(save_path,'mymodel_%s.csv'%(extra_cmt))).drop('Unnamed: 0',axis=1)                
                save_fig(leaderboard_df_sp,save_path,x='model',y = 'train',filename=extra_cmt)
                save_fig(leaderboard_df_sp,save_path,x='model',y = 'test',filename=extra_cmt)

                miss_all_df.to_csv(os.path.join(save_path, 'miss.csv'))

                for y_ in ('train','test'):
                    plt.figure(figsize=(10,5))
                    sns.boxplot(leaderboard_df.query('model== "Neural Net"'),x='condition',y=y_,hue='methods')
                    plt.xticks(rotation=-15)
                    plt.title(f'Neural Net {y_}set ROC')
                    plt.ylim(0,)
                    plt.savefig(os.path.join(save_path,'NN_%s_%i.pdf'%(y_,iter)),bbox_inches='tight')
            except:
                print('skip saving file')
        # sys.exit()  #only run once
        

def clr_result_to_DF(results):
    '''
    transfer the clr_compare results to Dataframe.
    output will be column with ['method','recall train','recall train std','recall test']
    '''
    results_df = pd.DataFrame(results).T
    results_df = results_df.reset_index()
    if results_df.shape[1] == 3:
        results_df.columns=['model','train','test']
    elif results_df.shape[1] == 2:
        results_df.columns=['model','train']

    return results_df


def mergelevel(df):
    '''
    return df.columns
    use as df.columns = droplevel(df)
    '''
    return df.columns.map(lambda tup: '|'.join(map(str, tup)))

def quick_sampling(df,ratio=0.4,seed=None):
    '''
    input is df, return list of material name
    '''
    mtr = df['Material'].unique()
    iso = []
    ani = []
    for m in mtr:
        i_label = df.query('Material=="%s"'%m)['isotropy label'].iloc[0]       
        if i_label == 0:
            iso.append(m)
        elif i_label == 1:
            ani.append(m)
    t1,t2 = randsamp(0,len(iso),round(ratio*len(iso)),seed) #picked, rest
    t3,t4 = randsamp(0,len(ani),round(ratio*len(ani)),seed+seed)

    train_material = [iso[x] for x in t2] + [ani[x] for x in t4]
    test_material =[iso[x] for x in t1] + [ani[x] for x in t3]
    return train_material,test_material
