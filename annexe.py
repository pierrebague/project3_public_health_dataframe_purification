import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import plotly.express as px
# import plotly.graph_objs as go
import seaborn as sns
FICHIER ='C:/Users/pierr/Documents/OC/projet3/en.openfoodfacts.org.products.csv'

USEFULL_COLUMNS = ['code','url','product','quantity','packaging','brands','categories','purchase_places','countries','ingredients_text','allergens','traces','pnns','score','main_category',"_100g"] 
PARASITE_COLUMNS = ['serving_quantity']
MORE_HUNDRED_COLUMNS = ["energy_100g","energy-kj_100g","energy-kcal_100g","energy-from-fat_100g","carbon-footprint_100g","carbon-footprint-from-meat-or-fish_100g"]
CAN_BE_HUNDRED_COLUMNS =["fat_100g","saturated-fat_100g","monounsaturated-fat_100g","polyunsaturated-fat_100g","salt_100g","sodium_100g","sugars_100g","-sucrose_100g","-glucose_100g","-fructose_100g","-lactose_100g","-maltose_100g","polyols_100g","starch_100g","carbohydrates_100g","alcohol_100g","fruits-vegetables-nuts_100g","fruits-vegetables-nuts-dried_100g","fruits-vegetables-nuts-estimate_100g","collagen-meat-protein-ratio_100g","cocoa_100g"]
CAN_BE_NEGATIF_COLUMNS = ["nutrition-score-fr_100g","nutrition-score-uk_100g","ecoscore_score_fr"]
STRING_VARIABLE_COLUMNS = ['pnns_groups_1','pnns_groups_2','nutriscore_grade','ecoscore_grade_fr']

#open the csv file, take the columns corresponding to the pattern in USEFULL_COLUMNS_PATTERN variable and return it without duplicates
def open_csv_usefull_column():
    dataframe = pd.read_csv(FICHIER,sep='\t',low_memory=False)
    print("Ouverture initiale, fichier avec",dataframe.shape[1],"variables et",dataframe.shape[0],"produits")
    column_list = []
    for reg in USEFULL_COLUMNS:
        regex = re.compile(".*" + reg + ".*")
        column_list = column_list + list(filter(regex.match, dataframe.columns))
    dataframe.drop(dataframe.columns.difference(column_list),1,inplace=True)
    dataframe.drop(PARASITE_COLUMNS,1,inplace=True)
    dataframe.dropna(axis = 'columns', how = 'all',inplace=True)
    dataframe.drop_duplicates(subset=column_list.remove('code'),inplace=True)
    print("Nettoyage initial, fichier avec",dataframe.shape[1],"produits et",dataframe.shape[0],"variables")
    return dataframe

# barplot the logarithm of the number of values for each variable in the dataframe
def barplot_logarithm(dataframe,title):
    fig = sns.barplot(x=dataframe.sort_values(ascending=False).index, y= np.log10(dataframe.sort_values(ascending=False)))
    fig.set_title(title)
    fig.axes.xaxis.set_visible(False)
    
# plot an histogram with minimum dataframe variables in different interval
def barplot_number_of_values(dataframe):
    min_list = dataframe.min()
    values_tab = ["[-∞,-100]","]-100,-10]","]-10,-1]","]-1,0[","[0]","]0,1[","[1,10[","[10,100[","[100,+∞]"]
    interval_sum = [0,0,0,0,0,0,0,0,0]
    for value in min_list:
        if value == 0:
            interval_sum[4] += 1
        elif abs(value) >= 100:
            interval_sum[0 if value < 0 else 8] += 1
        elif abs(value) >= 10:
            interval_sum[1 if value < 0 else 7] += 1  
        elif abs(value) >= 1:
            interval_sum[2 if value < 0 else 6] += 1
        else:
            interval_sum[3 if value < 0 else 5] += 1
    f, ax = plt.subplots(1, 1, figsize=(10, 5))
    sns.barplot(x=interval_sum, y=values_tab,ax=ax)       
    ax.set_title('minimum dataframe variables',fontsize=18)
    ax.set_xlabel('numbers of minimums')
    ax.set_ylabel('minimum value of variables')
    ax.xaxis.tick_top()     

# return the good value quantile depending of the number of value in the column
def find_good_pourcent_quantile(dataframe,column):
    count = dataframe[column].count()
    if count > 1000000:
        return 0.0001
    elif count > 100000:
        return 0.001
    else:
        return 0.01     


# remove a sublist of a list
def remove_list_elements(big_list,elements_list):
    big_list = [elem for elem in big_list if elem not in elements_list]
    return big_list

# drop quantile part in the dataframe for each numerical column depends on their type of values   
def remove_little_quantile(entry_dataframe):
    general_regex = re.compile(".*_100g.*")
    general_column_list = list(filter(general_regex.match, entry_dataframe.columns))
    general_column_list = remove_list_elements(general_column_list,MORE_HUNDRED_COLUMNS)
    general_column_list = remove_list_elements(general_column_list,CAN_BE_HUNDRED_COLUMNS)
    general_column_list = remove_list_elements(general_column_list,CAN_BE_NEGATIF_COLUMNS)
    mult_param = 3
    mult_param_energy = 3
    dataframe = entry_dataframe.copy()
    print("forme du dataframe d'entré :",dataframe.shape[0]," lignes et ",dataframe.shape[1]," colonnes.")
    
    for column in CAN_BE_NEGATIF_COLUMNS:        
        if column not in dataframe.columns:
            break
        value_count_entrance = dataframe[column].shape[0]     
        
        if len(dataframe[column].value_counts()) > 0:
            pourcent_quantile = find_good_pourcent_quantile(dataframe,column)            
            try:
                quantile_min = np.nanquantile(dataframe[column].to_numpy(),pourcent_quantile)
            except ValueError:
                break            
            try:    
                quantile_max = np.nanquantile(dataframe[column].to_numpy(),1-pourcent_quantile)
            except ValueError:
                break    
            mean = dataframe[column].mean()                
            try:
                dataframe.drop(dataframe[((dataframe[column] < quantile_min) & (abs(dataframe[column] - mean) > mult_param_energy * mean ))].index,inplace=True)
            except ValueError:
                print(column,": Pas de valeurs à enlever pour le quantile inferieur")            
            try:
                dataframe.drop(dataframe[((dataframe[column] > quantile_max) & (abs(dataframe[column] - mean) > mult_param_energy * mean))].index,inplace=True)
            except ValueError:
                print(column,": Pas de valeurs à enlever pour le quantile superieur")
            
            print("column :",column,", ",value_count_entrance - dataframe[column].shape[0]," values erased")
        
        else:
            print("Colonne vide : ",column)
            dataframe.drop(column,axis = 'columns',inplace=True)
    
    for column in MORE_HUNDRED_COLUMNS:        
        value_count_entrance = dataframe[column].shape[0]
        if len(dataframe[column].value_counts()) > 0:
            pourcent_quantile = find_good_pourcent_quantile(dataframe,column)
            try:
                quantile_min = np.nanquantile(dataframe[column].to_numpy(),pourcent_quantile)
                quantile_max = np.nanquantile(dataframe[column].to_numpy(),0.99)
                median = dataframe[column].median()
                dataframe.drop(dataframe[(((dataframe[column] < quantile_min) & (abs(dataframe[column] - median) > mult_param_energy * median )) | (dataframe[column] < 0))].index,inplace=True)
            except ValueError:
                print(column,": Pas de valeurs à enlever pour le quantile inferieur")
            try:
                dataframe.drop(dataframe[(((dataframe[column] > quantile_max) & (abs(dataframe[column] - median) > mult_param_energy * median)))].index,inplace=True)
            except ValueError:
                print(column,": Pas de valeurs à enlever pour le quantile superieur")
            print("column :",column,", ",value_count_entrance - dataframe[column].shape[0]," values erased")
        else:
            print("Colonne vide : ",column)
            dataframe.drop(column,axis = 'columns',inplace=True)
    
    
    for column in general_column_list:
        value_count_entrance = dataframe[column].shape[0]
        if len(dataframe[column].value_counts()) > 0:
            pourcent_quantile = find_good_pourcent_quantile(dataframe,column)
            try:
                quantile_min = np.nanquantile(dataframe[column].to_numpy(),pourcent_quantile)
                quantile_max = np.nanquantile(dataframe[column].to_numpy(),1-pourcent_quantile)
                mean = dataframe[column].mean()
                dataframe.drop(dataframe[(((dataframe[column] < quantile_min) & (abs(dataframe[column] - mean) > mult_param * mean)) | (dataframe[column] < 0))].index,inplace=True)
            except ValueError:
                print(column,": Pas de valeurs à enlever pour le quantile inferieur")
            try:
                dataframe.drop(dataframe[(((dataframe[column] > quantile_max) & (abs(dataframe[column] - mean) > mult_param * mean)) | (dataframe[column] > 100))].index,inplace=True)
            except ValueError:
                print(column,": Pas de valeurs à enlever pour le quantile superieur ")
            print("column :",column,", ",value_count_entrance - dataframe[column].shape[0]," values erased")
        else:
            print("Colonne vide : ",column)
            dataframe.drop(column,axis = 'columns',inplace=True)
    
    for column in CAN_BE_HUNDRED_COLUMNS:
        value_count_entrance = dataframe[column].shape[0]
        if len(dataframe[column].value_counts()) > 0:
            try:
                dataframe.drop(dataframe[dataframe[column] < 0].index,inplace=True)
            except ValueError:
                print(column,": Pas de valeurs à enlever pour le quantile inferieur")
            try:
                dataframe.drop(dataframe[dataframe[column] > 100].index,inplace=True)
            except ValueError:
                print(column,": Pas de valeurs à enlever pour le quantile superieur")
            print("column :",column,", ",value_count_entrance - dataframe[column].shape[0]," values erased")
        else:
            print("Colonne vide : ",column)
            dataframe.drop(column,axis = 'columns',inplace=True)
    dataframe = dataframe.dropna(axis = 'columns', how = 'all')
    differences_column = entry_dataframe.columns.difference(dataframe.columns)
    print("forme du dataframe de sortie :",dataframe.shape[0]," lignes et ",dataframe.shape[1]," colonnes.")
    print("colonnes enlevé :",differences_column)
    return dataframe
    
# replace in the dataframe hyphen by space and lowercase all the text in the column_list given 
def text_regularisation(dataframe,column_list):
    for column in column_list:
        dataframe[column] = dataframe[column].str.lower()
        dataframe[column] = dataframe[column].str.replace('-',' ')
        if column == "pnns_groups_2":
            dataframe[column].replace("pizza pies and quiches","pizza pies and quiche",inplace = True)    

# plot an interactive heatmap of correlation dataframe  
def plot_heat_map(correlation_dataframe):
    datalist = correlation_dataframe.values.tolist()
    fig = px.imshow(np.array(datalist),
                    #labels=dict(x="variable 1",y="varible 2"),
                   x=correlation_dataframe.columns,
                   y=correlation_dataframe.columns,
                   width=700,
                   height=700)
    fig.update_layout(title_text="Correlation matrix",title_x = 0.6,title_y=0.95,title_font_size=25)
    fig.show()      
    
# return couples of variables where correlation score is superior to wanted_correlation from a correlation tab
def variables_correlation_over_parameter(correlation_dataframe,wanted_correlation_min,wanted_correlation_max):
    columns = correlation_dataframe.columns.tolist()
    print("Correlation entre",wanted_correlation_min,"et",wanted_correlation_max)
    for column1 in columns:
        for column2 in columns[columns.index(column1) + 1:]:
            if correlation_dataframe[column1][column2] >= wanted_correlation_min and correlation_dataframe[column1][column2] <= wanted_correlation_max:
                print(column1,"et",column2)

# say if the correlation of 1 is good or bad                
def good_or_bad_correlation(correlation_matrix,dataframe_correlate):
    for num,column in enumerate(correlation_matrix.columns):
        for row1 in correlation_matrix.columns[num+1:]:
            comptnan = 0
            comptvalues  = 0
            if correlation_matrix[column][row1] >= 1:
                print("-------------------------------------------------------------------------------")
                if ((dataframe_correlate[column].notna()) & (dataframe_correlate[row1].notna())).sum() > 25:
                    display(dataframe_correlate[((dataframe_correlate[column].notna()) & (dataframe_correlate[row1].notna()))][[column,row1]].head(5))
                    display(dataframe_correlate[((dataframe_correlate[column].isna()) & (dataframe_correlate[row1].notna()))][[column,row1]].head(5))
                    display(dataframe_correlate[((dataframe_correlate[column].notna()) & (dataframe_correlate[row1].isna()))][[column,row1]].head(5))
                    
                else:
                    print("Fausse correlation de 1 pour ",column," and ",row1)   

# bivariate plot of two elments
def bivariate_plot_multiple_elements(dataframe,variable1,variable2,title):
    f, ax = plt.subplots(figsize=(6, 6))
    sns.set_theme(style="dark")
    sns.scatterplot(x=dataframe[variable1], y=dataframe[variable2], s=5, color=".15")
    sns.histplot(x=dataframe[variable1], y=dataframe[variable2], bins=50, pthresh=.1, cmap="mako")
    sns.kdeplot(x=dataframe[variable1], y=dataframe[variable2], levels=5, color="w", linewidths=1)
    plt.title(title, fontsize=16)    

# plot density correlation with marginals histograms    
def smooth_density_marginal_histograms(dataframe,x_values,y_values,xmax,ymax,title):
    sns.set_theme(style="white")
    g = sns.JointGrid(data=dataframe, x=x_values, y=y_values, space=0)
    g.plot_joint(sns.kdeplot,
                 fill=True, clip=((0, xmax), (0, ymax)),
                 thresh=0, levels=100, cmap="mako")
    s = g.plot_marginals(sns.histplot, color="#03051A", alpha=1, bins=25)
    plt.title(title, fontsize=16,x=-2.5, y=-0.2)
    
    
# function to find pnns group 1 of all product who contain "part"
def find_pnns1_part(dataframe,part):
    all_needed = dataframe.loc[dataframe[part].dropna().index]
    print(all_needed.shape[0],"produits trouvés")
    plt.pie(all_needed['pnns_groups_1'].value_counts(),labels = all_needed['pnns_groups_1'].value_counts().index)
    all_the_title = "pnns group for food containing "+part[:-5]
    text = plt.title(all_the_title, fontsize=15)

# function to find pnns group 2 of all product who contain "part"    
def find_pnns2_part(dataframe,column):
    all_needed = dataframe.loc[dataframe[column].dropna().index]
    print(all_needed.shape[0],"produits trouvés")
    plt.pie(all_needed['pnns_groups_2'].value_counts(),labels = all_needed['pnns_groups_2'].value_counts().index)
    all_the_title = "pnns group for food containing "+column[:-5]
    text = plt.title(all_the_title, fontsize=15)

