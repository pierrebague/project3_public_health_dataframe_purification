import numpy as np
import pandas as pd
import re
# import matplotlib.pyplot as plt
# import plotly.graph_objs as go
# import seaborn as sns
FICHIER ='F:/cour/OC/projet3/en.openfoodfacts.org.products.csv'

USEFULL_COLUMNS_PATTERN = ['code','url','product_name','quantity','packaging','packaging_tags','packaging_text','brands','brands_tags','categories','categories_tags','categories_en','purchase_places','stores','countries','countries_tags','countries_en','ingredients_text','allergens','allergens_en','traces','traces_tags','traces_en','no_nutriments','nutriscore_score','nutriscore_grade','pnns_groups_1','pnns_groups_2','ecoscore_score_fr','ecoscore_grade_fr','main_category','main_category_en','image_url','_100g']
more_hundred_columns = ["energy-kj_100g","energy-kcal_100g","energy_100g","energy-from-fat_100g","carbon-footprint_100g","carbon-footprint-from-meat-or-fish_100g"]
can_be_hundred_columns =["fat_100g","saturated-fat_100g","monounsaturated-fat_100g","polyunsaturated-fat_100g"]
can_be_negatif_columns = ["nutrition-score-fr_100g","nutrition-score-uk_100g"]

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
def remove_little_quantile3(entry_dataframe):
    general_regex = re.compile(".*_100g.*")
    general_column_list = list(filter(general_regex.match, entry_dataframe.columns))
    general_column_list = remove_list_elements(general_column_list,more_hundred_columns)
    general_column_list = remove_list_elements(general_column_list,can_be_hundred_columns)
    general_column_list = remove_list_elements(general_column_list,can_be_negatif_columns)
    mult_param = 3
    mult_param_energy = 5
    dataframe = entry_dataframe.copy()
    print("forme du dataframe d'entré :",dataframe.shape[0]," lignes et ",dataframe.shape[1]," colonnes.")
    
    for column in can_be_negatif_columns:        
        value_count_entrance = dataframe[column].shape[0]
        if len(dataframe[column].value_counts()) > 0:
            pourcent_quantile = find_good_pourcent_quantile(dataframe,column)
            try:
                quantile_min = np.nanquantile(dataframe[column].to_numpy(),pourcent_quantile)
                quantile_max = np.nanquantile(dataframe[column].to_numpy(),1-pourcent_quantile)
                mean = dataframe[column].mean()
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
    
    for column in more_hundred_columns:        
        value_count_entrance = dataframe[column].shape[0]
        if len(dataframe[column].value_counts()) > 0:
            pourcent_quantile = find_good_pourcent_quantile(dataframe,column)
            try:
                quantile_min = np.nanquantile(dataframe[column].to_numpy(),pourcent_quantile)
                quantile_max = np.nanquantile(dataframe[column].to_numpy(),1-pourcent_quantile)
                mean = dataframe[column].mean()
                dataframe.drop(dataframe[(((dataframe[column] < quantile_min) & (abs(dataframe[column] - mean) > mult_param_energy * mean )) | (dataframe[column] < 0))].index,inplace=True)
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
    
    for column in can_be_hundred_columns:
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

#open the csv file, take the columns corresponding to the pattern in USEFULL_COLUMNS_PATTERN variable and return it
def open_csv_usefull_column():
    dataframe = pd.read_csv(FICHIER,sep='\t',low_memory=False)
    column_list = []
    for pattern in USEFULL_COLUMNS_PATTERN:
        regex = re.compile(".*" + pattern + ".*")
        column_list = column_list + list(filter(regex.match, dataframe.columns))
    dataframe.drop(dataframe.columns.difference(column_list),1,inplace=True)
    dataframe = dataframe.dropna(axis = 'columns', how = 'all')
    return dataframe

# remove row with values under the 1% quintile and over the 99% quintile of a dataframe list of column
# remove aberant values
# nanquantile
def remove_little_quantile(dataframe,column_list):
    for column in column_list:
#         print(column)
        if len(dataframe[column].value_counts()) > 0:
#         display(dataframe.head(5))
#         dataframe.drop(dataframe[dataframe[column].apply(lambda x: not x.isnumeric())].index,inplace=True)
#         dataframe[column] = pd.to_numeric(dataframe[column])
            quantile_min = np.nanquantile(dataframe[column].to_numpy(),0.01)
            quantile_max = np.nanquantile(dataframe[column].to_numpy(),0.99)
            dataframe.drop(dataframe[dataframe[column] < quantile_min].index,inplace=True)
            dataframe.drop(dataframe[dataframe[column] > quantile_max].index,inplace=True)
        else:
            dataframe.drop(column,axis = 'columns',inplace=True)
    dataframe = dataframe.dropna(axis = 'columns', how = 'all')
    
def remove_little_quantile2(entry_dataframe,column_list):
    mult_param = 3
    dataframe = entry_dataframe.copy()
    for column in column_list:
#         print(column)
        if len(dataframe[column].value_counts()) > 0:
#         display(dataframe.head(5))
#         dataframe.drop(dataframe[dataframe[column].apply(lambda x: not x.isnumeric())].index,inplace=True)
#         dataframe[column] = pd.to_numeric(dataframe[column])
            quantile_min = np.nanquantile(dataframe[column].to_numpy(),0.01)
            quantile_max = np.nanquantile(dataframe[column].to_numpy(),0.99)
            mean = dataframe[column].mean()
            dataframe.drop(dataframe[(((dataframe[column] < quantile_min) & (abs(dataframe[column] - mean) > mult_param * mean)) | (dataframe[column] < 0))].index,inplace=True)
            dataframe.drop(dataframe[(((dataframe[column] > quantile_max) & (abs(dataframe[column] - mean) > mult_param * mean)) | (dataframe[column] > 100))].index,inplace=True)
        else:
            dataframe.drop(column,axis = 'columns',inplace=True)
    dataframe = dataframe.dropna(axis = 'columns', how = 'all')
    differences_column = entry_dataframe.columns.difference(dataframe.columns)
    print("colonnes enlevé :",differences_column)
    return dataframe

# replace in the dataframe hyphen by space and lowercase all the text in the column_list given 
def text_regularisation(dataframe,column_list):
    for column in column_list:
        dataframe[column] = dataframe[column].str.lower()
        dataframe[column] = dataframe[column].str.replace('-',' ')
        
# return couples of variables where correlation score is superior to wanted_correlation from a correlation tab
def variables_correlation_over_parameter(correlation_dataframe,wanted_correlation):
    columns = correlation_dataframe.columns.tolist()
    for column1 in columns:
        for column2 in columns[columns.index(column1) + 1:]:
            if correlation_dataframe[column1][column2] >= wanted_correlation:
                print(column1,column2)

                
# FICHIERS = ["EdStatsCountry.csv","EdStatsCountry-Series.csv","EdStatsData.csv","EdStatsFootNote.csv"
# ,"EdStatsSeries.csv"]
# LOCALISATION ='F:/cour/OC/projet2/'
# INDEX = ["secondary","tertiary","school|educationnal","student","inhabitant|household","population","technology|computer|internet"]
# VALUES_NOT_WANTED = ["WLD","ARE","LMC","LIC","LMY","UMC","MIC","HIC","NOC","OEC","EUU","EAS","EAP","SAS","OED","ECS","LCN","LAC","LDC","SSF","SSA","ECA","MEA","NAC","HPC","MNA","EMU","ARB","IDN","ZAF"]
# COLOR_LIST = ["pastel","muted","colorblind","deep","dark","bright"]
# NOT_IN_STUDY_YEARS = ['FIN','NZL','ISL','AUT','SMR','CAN']

# def open_csv(num_fichier,index_column="",column_list=""):
    # if index_column == "" and column_list == "":
        # fichier_lu = pd.read_csv(LOCALISATION + FICHIERS[num_fichier])
        # fichier_lu = fichier_lu.dropna(axis = 'columns', how = 'all')
    # elif index_column != "" and column_list != "":
        # fichier_lu = pd.read_csv(LOCALISATION + FICHIERS[num_fichier],index_col=index_column,names=column_list)
        # fichier_lu = fichier_lu.dropna(axis = 'columns', how = 'all')
    # elif index_column == "" and column_list != "":
        # fichier_lu = pd.read_csv(LOCALISATION + FICHIERS[num_fichier],names=column_list)
        # fichier_lu = fichier_lu.dropna(axis = 'columns', how = 'all')
    # else:
        # fichier_lu = pd.read_csv(LOCALISATION + FICHIERS[num_fichier],index_col=index_column)
        # fichier_lu = fichier_lu.dropna(axis = 'columns', how = 'all')
    # print("\n",FICHIERS[num_fichier], " is open")
    # print("fichier avec ",fichier_lu.shape[0]," lignes et ",fichier_lu.shape[1]," colonnes.")
    # return fichier_lu

# def print_empty_stats(dataframe):
    # data_number = dataframe.count(axis='index').cumsum().tolist()
    # pourcentage_values = int(data_number[-1]) / (int(dataframe.shape[0]) * int(dataframe.shape[1])) * 100
    # print("le dataframe est rempli à ",format(pourcentage_values,".2f"),"%\n")

# def print_samples(data_frame,number_of_rows):
    # display(data_frame.sample(number_of_rows,random_state = 148625))

# def index_str_contains(index,dataframe,regex_var):
    # new_index = index.str.contains(regex_var,case=False,regex=True,na=False)
    # return new_index

# def indicator_name_list(dataframe):
    # index = dataframe['Indicator Name'].squeeze()
    # variable_list = []
    # for regex in INDEX:
        # index_temp = index_str_contains(index,dataframe,regex)
        # set_temp = set(dataframe[index_temp]['Indicator Name'].unique())
        # for variable in variable_list:
            # set_temp = set_temp - variable
        # print("Pour le regex ",regex," : ",len(set_temp)," variables de trouvé")    
        # variable_list.append(set_temp)
    # return variable_list

# def to_keep(dataframe,columns_to_keep):
    # reduct_dataframe = open_csv(dataframe)
    # for column in reduct_dataframe.columns:
        # if column not in columns_to_keep:
            # reduct_dataframe = reduct_dataframe.drop([column],axis = 1)
    # print("nouveau format du fichier : ",reduct_dataframe.shape[0]," lignes et ",reduct_dataframe.shape[1]," colonnes.")
    # return reduct_dataframe

# def take_needed_rows(dataframe,list_values):
    # new_dataframe = pd.DataFrame([])
    # for value in list_values:
        # new_dataframe = pd.concat([new_dataframe,dataframe.loc[dataframe['Indicator Name'] == value]])
    # return new_dataframe

# def replace_ESC(dataframe, value_or_number=0):
    # if value_or_number == 0:
        # new_dataframe = dataframe.replace(["High income: nonOECD","Upper middle income","Lower middle income","High income: OECD","Low income"],["High","Upper \nmiddle","Lower \nmiddle","High","Low"])
    # else:
        # new_dataframe = dataframe.replace(["High income: nonOECD","Upper middle income","Lower middle income","High income: OECD","Low income"],[5,4,2,5,1])
    # return new_dataframe

# def sns_graph(fichierESC3):
    # sns.set(font_scale=5)
    # sns.set_theme(style="darkgrid")
    # ax = sns.countplot(x="Income Group",order = ["High","Upper \nmiddle","Lower \nmiddle","Low"],\
                   # data = fichierESC3,palette=["tab:red","tab:orange","cornflowerblue","darkblue",]).\
                    # set_title("Numbers of countries by income group")

# def choropleth_map(dataframe,titre,index=False,year='2001',column='Income Group'):
    # if index:
        # countries = dataframe.index.tolist()
        # z = dataframe[year].tolist()
        # titre = titre + year
    # elif not index:
        # countries = dataframe['Country Code'].tolist()
        # z = dataframe[column].tolist()
    # layout = dict(geo={'scope': 'world'})
    # scl = [[0.0, 'darkblue'],[0.2, 'cornflowerblue'],[0.4, 'cornflowerblue'],\
               # [0.6, 'orange'],[0.8, 'orange'],[1.0, 'red']]
    # data = dict(
        # type='choropleth',
        # locations=countries,
        # locationmode='ISO-3',
        # colorscale=scl,
        # autocolorscale = False,
        # marker = dict(line = dict (color = 'rgb(0,0,0)', width = 1)),z=z)
    # map = go.Figure(data=[data], layout=layout)
    # map.update_layout(
    # title={
        # 'text': titre,
        # 'y':0.9,
        # 'x':0.5,
        # 'xanchor': 'center',
        # 'yanchor': 'top'
        # },    
        # title_font_size=30)
    # map.show()

# def ratio_epuration(data_frame,ratio):
    # nb_lignes = data_frame.shape[0]
    # tab_isna = data_frame.isna().sum()
    # list_index = tab_isna.index.tolist()
    # df_epure = pd.DataFrame()
    # for index,value in enumerate(tab_isna):
        # if value <= nb_lignes * (1 - ratio):
            # df_epure[list_index[index]] = data_frame[list_index[index]]
    # return df_epure

# def clean_data(dataframe,ratio):
    # dataframe2 = dataframe.replace(0,np.nan)
    # dataframe3 = dataframe2.dropna(axis = 'columns', how = 'all')
    # dataframe4 = ratio_epuration(dataframe3,ratio)
    # return dataframe4

# def create_range(dataframe,quantity_print,which_one=-1):
    # if quantity_print == 1:
        # if which_one == 0:
            # return [dataframe.columns[0]]
        # elif which_one == -1:
            # return [dataframe.columns[-1]]
        # else:
            # try:
                # dataframe[str(which_one)]
            # except ValueError:
                # print("Non valid column")
    # else:
        # last_elem = int(dataframe.columns[-1])
        # column_nbr = int(len(dataframe.columns))
       
        # if column_nbr % (quantity_print - 1) == 0:
            # range_step = int(column_nbr / quantity_print)
        # else:
            # range_step = int(column_nbr / (quantity_print - 1))
        # begin_year = last_elem
        # for step in range(quantity_print-1):
            # begin_year -= range_step
        # return range(begin_year,last_elem+1,range_step)

# def most_filled_column(dataframe):
    # mini = dataframe[dataframe.columns[-1]].isna().sum()
    # column_mini = dataframe.columns[-1]
    # for column in reversed(dataframe.columns):
        # isna_sum = dataframe[column].isna().sum()
        # if mini > isna_sum:
            # mini = isna_sum
            # column_mini = column
    # return column_mini


# def fill_dataframe(dataframe):
    # return dataframe.replace(0,np.nan).transpose().fillna(method='ffill').transpose()

# def sort_dataframe(dataframe,sort_year=''):
    # dataframe2 = fill_dataframe(dataframe)
    # if sort_year=='':
        # best_column_to_sort = most_filled_column(dataframe)
    # else:
        # best_column_to_sort = sort_year 
        # for code in VALUES_NOT_WANTED:
            # try:
                # dataframe2 = dataframe2.drop([code],axis = 0)
            # except:
                # pass
    # dataframe2 = dataframe2.sort_values(by=[best_column_to_sort],ascending =False)
    # return dataframe2

# def print_top_values(dataframe,title,value1,value2,sort_year=''):
    # dataframe2 = sort_dataframe(dataframe,sort_year)
    # if value1 == 0:
        # dataframe3 = dataframe2.head(value2).transpose()
        # title = "Top " + str(value2) + " " + title
    # else:
        # dataframe3 = dataframe2.head(value2).tail(value2 - value1 + 1).transpose()
        # title = "Top " + str(value1) + " to " + str(value2) + " " + title
    # lines = dataframe3.plot.line().set_title(title)

# def last_value(dataframe,new_column):
    # dataframe2= dataframe.copy()
    # dataframe2.dropna(axis = 'columns', how = 'all',inplace=True)
    # dataframe2[new_column] = np.nan
    # dataframe2 = dataframe2.replace(0,np.nan)
    # dataframe2.transpose().fillna(method='ffill',inplace=True)
    # dataframe2.drop(dataframe2.columns.difference([new_column]),1,inplace=True)    
    # for code in VALUES_NOT_WANTED:
            # try:
                # dataframe2 = dataframe2.drop([code],axis = 0)
            # except:
                # pass
    # return dataframe2

# def rank_dataframe(dataframe,new_column):
    # dataframe2 = last_value(dataframe,new_column)
    # dataframe2 = dataframe2.sort_values(by=new_column,ascending=False)
    # maxi = float(dataframe2.iloc[0])
    # part = maxi/4
    # part2 = part
    # part3 = part*2
    # part4 = part*3
    # for row in range(dataframe2.shape[0]):
        # if float(dataframe2.iloc[row]) < part2:
            # dataframe2.iloc[row] = int(1)
        # elif float(dataframe2.iloc[row]) < part3:
            # dataframe2.iloc[row] = int(2)
        # elif float(dataframe2.iloc[row]) < part4:
            # dataframe2.iloc[row] = int(3)
        # else:
            # dataframe2.iloc[row] = int(4)
    # return dataframe2.astype(int)

# def horizontal_bar_plot_tri(dataframe):
    # sns.set_theme(style="whitegrid")
    # f, ax = plt.subplots(figsize=(6, 15))
    # dataframe2 = dataframe.sort_values(by=["Income Group","Internet","Computer"],ascending=False)
    # sns.set_color_codes("colorblind")
    # sns.barplot(x="Computer2", y="Country Code", data=dataframe2,label="Computer owner rank", color="b")
    # sns.set_color_codes("muted")
    # sns.barplot(x="Internet2", y="Country Code", data=dataframe2,label="Internet user rank", color="b")
    # sns.set_color_codes("pastel")
    # sns.barplot(x="Income Group", y="Country Code", data=dataframe2,label="Income Group rank", color="b")
    # ax.legend(ncol=1, loc="lower right", frameon=True,fontsize='large')
    # plt.xlabel( xlabel="Scoring by country",fontsize=18)
    # ax.set(xlim=(0, 15), ylabel="", xlabel="Scoring by country")
    # sns.despine(left=True, bottom=True)


# def horizontal_bar_plot_mono(dataframe,sort_by,title,xmin,xmax):
    # sns.set_theme(style="whitegrid")
    # f, ax = plt.subplots(figsize=(6, 15))
    # dataframe2 = dataframe.sort_values(by=[sort_by],ascending=False)
    # sns.set_color_codes("pastel")
    # sns.barplot(x=sort_by, y=dataframe2.index, data=dataframe2,label=sort_by, color="b")
    # ax.legend(ncol=1, loc="lower right", frameon=True,fontsize='large')
    # ax.xaxis.tick_top()    
    # if title == "Study years in selected countries":
        # for i,value in enumerate(dataframe2[sort_by]):
            # ax.text(value+3/xmin, i + 0.2,str(value),fontsize=15)
    # else:
        # for i,value in enumerate(dataframe2[sort_by]):
            # ax.text(value+3/xmin, i + 0.2,str(int(value)),fontsize=15)    
    # plt.xlabel( xlabel=title,fontsize=18)
    # ax.set(xlim=(xmin, xmax), ylabel="", xlabel=title)
    # sns.despine(left=True, bottom=True)
    

# def top_countries_with_data(dataframe):
    # dataframe2 = dataframe.copy()
    # for country in NOT_IN_STUDY_YEARS:
        # dataframe2.drop(dataframe2[dataframe2["Country Code"] == country].index,inplace =True)
    # return dataframe2

# def potential_years_study(dataframe1,dataframe2,selected_countries):          
    # dataframe = dataframe1.join(dataframe2,how='outer')
    # dataframe.fillna(1,inplace=True)
    # multiple_row = len(dataframe2.columns)
    # new_col_list = []
    # if multiple_row>1:
        # for column in range(len(dataframe2.columns)):
            # new_col = "potential_"+dataframe2.columns[column][-4:]
            # new_col_list.append(new_col)
            # dataframe[new_col] = dataframe[dataframe.columns[0]] * dataframe[dataframe.columns[column +1]]
    # else:
        # dataframe["potential"] = dataframe[dataframe.columns[0]] * dataframe[dataframe.columns[1]]
    # dataframe = dataframe.loc[selected_countries,:]    
    # if multiple_row>1:
        # return dataframe.sort_values(by=[new_col_list[0]],ascending=False)
    # else:
        # return dataframe.sort_values(by=['potential'],ascending=False)

# def take_value(dataframe,new_column,years):
    # dataframe2= dataframe.copy()
    # dataframe2.dropna(axis = 'columns', how = 'all',inplace=True)
    # dataframe2 = dataframe2.replace(0,np.nan)
    # dataframe2.transpose().fillna(method='ffill',inplace=True)
    # dataframe2.drop(dataframe2.columns.difference(years),1,inplace=True)
    # for year in years:
        # dataframe2= dataframe2.rename(columns={year:new_column+"_"+year})
    # for code in VALUES_NOT_WANTED:
            # try:
                # dataframe2 = dataframe2.drop([code],axis = 0)
            # except:
                # pass
    # return dataframe2

# def transforme_for_scatterplot(dataframe):
    # df1 = dataframe.reset_index()
    # df11 = df1.drop(df1.columns.difference(["Country Code","prediction_new_students_2020"]),1)
    # df12 = df1.drop(df1.columns.difference(["Country Code","prediction_new_students_2025"]),1)
    # df13 = df1.drop(df1.columns.difference(["Country Code","prediction_new_students_2030"]),1)
    # df11["year"] = "students_number_2020"
    # df12["year"] = "students_number_2025"
    # df13["year"] = "students_number_2030"
    # df11.rename(columns={"prediction_new_students_2020":"students_number"},inplace=True)
    # df12.rename(columns={"prediction_new_students_2025":"students_number"},inplace=True)
    # df13.rename(columns={"prediction_new_students_2030":"students_number"},inplace=True)
    # return pd.concat([df11,df12,df13])

# def create_list_for_scatterplot(begin,end,time):
    # row_list = []
    # for mult in range(3):
        # for num in range(begin-1,end):            
            # row_list.append(int(mult*time+num))
    # return row_list

# def scatterplot_student_number(dataframe,title,mini,maxi):
    # size_df = int(len(dataframe)/3)
    # ax = plt.axes()
    # plt.title("Students number prediction in thousand")
    # fig = sns.scatterplot(data=dataframe.iloc[create_list_for_scatterplot(mini,maxi,size_df)], x="Country Code", y="students_number",hue="year")

# def create_list_for_scatterplot2(begin,end):
    # row_list = []
    # for num in range(begin-1,end):
        # for mult in range(3):
            # row_list.append(int(mult+3*num))
    # return row_list

# def display_potential_years_study(dataframe_study_year,final_df,mini,maxi):
    # final_df2 = final_df.copy()
    # final_df2["potential"] = final_df2["students_number"]
    # final_df3 = final_df2.merge(dataframe_study_year,left_on="Country Code",right_on=dataframe_study_year.index)
    # final_df3["potential"] = final_df3["potential"]*final_df3["study_years_expected"]
    # final_df3 = final_df3.replace("students_number_2020","potential_2020")
    # final_df3 = final_df3.replace("students_number_2025","potential_2025")
    # final_df3 = final_df3.replace("students_number_2030","potential_2030")
    # final_df3 = final_df3.sort_values(by=["potential"],ascending=False)
    # size_df = int(len(final_df3)/3)
    # ax = plt.axes()
    # plt.title("Countries potential")
    # fig = sns.scatterplot(data=final_df3.iloc[create_list_for_scatterplot2(mini,maxi)], x="Country Code", y="potential",hue="year")