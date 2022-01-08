


import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import numpy as np
import distython
from distython import HEOM
#Import TfIdfVectorizer from the scikit-learn library
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.neighbors import NearestNeighbors

#%%
from flask import Flask, redirect, url_for, request,render_template

#%%
app = Flask(__name__,template_folder='C:/Users/ADMIN/Documents/term 5/Data Science/template_folder/')
app._static_folder = "C:/Users/ADMIN/Documents/term 5/Data Science/"
#%%
@app.route('/success')
def success():
    scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name('sheet.json', scope)
    client = gspread.authorize(creds)
    sheet = client.open('Roommate').sheet1
    Responses = sheet.get_all_records()
#print(Responses)
    data= pd.DataFrame(Responses)
#data = data.drop([35])
    names = data['Full Name']
    Gender = data['Gender preferences of your roommate']
    numerized=data
    numerized['Room preferences'] = np.where(data['Room preferences'].apply(lambda x: 'and' in x), 1, 0)
    numerized['Gender'] = np.where(data['Gender'].apply(lambda x: 'Male' in x), 0, 1)
    numerized['Occupation'] = np.where(data['Occupation'].apply(lambda x: 'Student' in x), 1, 0)
    numerized['Are you a Vegetarian?'] = np.where(data['Are you a Vegetarian?'].apply(lambda x: 'Yes' in x), 1, 0)
    numerized['Grocery shopping'] = np.where(data['Grocery shopping'].apply(lambda x: 'Would keep grocery separate' in x), 0, 1)
    numerized['Do you smoke? (and smoke up?)'] = np.where(data['Do you smoke? (and smoke up?)'].apply(lambda x: 'Yes' in x), 1, 0)
    numerized['Do you drink?'] = np.where(data['Do you drink?'].apply(lambda x: 'Yes' in x), 1, 0)
    numerized['Is it cool with you if your roommate organizes house parties?'] = np.where(data['Is it cool with you if your roommate organizes house parties?'].apply(lambda x: 'Yes' in x), 1, 0)
    numerized['Okay last question! When are your eyes first wide open in the morning?'] = np.where(data['Okay last question! When are your eyes first wide open in the morning?'].apply(lambda x: 'Early morning (anytime before 7 am easily)' in x), 0, 1)
    numerized['Be honest, are you a procrastinator? '] = np.where(data['Be honest, are you a procrastinator? '].apply(lambda x: 'Yes' in x), 1, 0)

#numerized["House Cleaning and Cooking"].value_counts()
    num1 = {"House Cleaning and Cooking": {"Maid for both": 0, "Maid for Cooking only": 1,"Maid for none": 2,"Would like to keep maid for housecleaning": 3}}
    numerized.replace(num1, inplace=True)

#numerized["So what are your work timings?"].value_counts()
    num2 = {"So what are your work timings?": {"Day shift": 0, "Varies": 1,"Night shift": 2}}
    numerized.replace(num2, inplace=True)

#numerized["Gender preferences of your roommate"].value_counts()
    num3 = {"Gender preferences of your roommate": {"Male": 0, "Female": 1, "Indifferent": 2}}
    numerized.replace(num3, inplace=True)

#numerized["Say it's a Friday night, where would you most likely be?"].value_counts()
    num4 = {"Say it's a Friday night, where would you most likely be?": {"play Football or hit the gym": 0, "The Club! (obviously!)": 0, "Read a book or watch a movie curled up  like a burrito (ah, sounds nice just reading it!)": 1,"Work doesn't allow for such luxuries":1}}
    numerized.replace(num4, inplace=True)

#numerized["What's the first thing you do when you get out of bed?"].value_counts()
    num5 = {"What's the first thing you do when you get out of bed?": {"Go back to sleep, until it's too late to be acceptably late (we've all been there)": 0, "Make the bed": 1, "Start planning my day ahead": 1, "Knock over stuff on the floor":2}}
    numerized.replace(num5, inplace=True)

#numerized["Do you have a partner?"].value_counts()
    num6 = {"Do you have a partner?": {"No": 0, "Prefer not to say": 0, "Yes": 1}}
    numerized.replace(num6, inplace=True)

#numerized["So, how often does your partner visit?"].value_counts()
    num7 = {"So, how often does your partner visit?": {"Na": 0,"": 0, "Never": 1,"1-2 a week" : 1, "Rarely": 1,"once in a month or two": 2, "well, almost all the time, because they're the best human being on the planet!": 2}}
    numerized.replace(num7, inplace=True)

#numerized["Is it cool if your roommate's partner visits often?"].value_counts()
    num8 = {"Is it cool if your roommate's partner visits often?": {"No": 0, "Yes": 1}}
    numerized.replace(num8, inplace=True)

#numerized["What is your budget? (per person)"].value_counts()
    num9 = {"What is your budget? (per person)": {"4000-6000": 0, "6000-8000": 1,"8000-10000":2,"10000-12000":3}}
    numerized.replace(num9, inplace=True)
#numerized
    h=pd.DataFrame(numerized['Room preferences'])
    heomdat=pd.concat([h,data['Full Name'],data['Email Address'],numerized['Gender'],numerized['Occupation'],
           numerized['Are you a Vegetarian?'],numerized['Grocery shopping'],
           numerized['Do you smoke? (and smoke up?)'],numerized['Do you drink?'],
           numerized['Is it cool with you if your roommate organizes house parties?'],
           numerized['Okay last question! When are your eyes first wide open in the morning?'],
           numerized['Be honest, are you a procrastinator? '],
           numerized["House Cleaning and Cooking"],
           numerized["So what are your work timings?"],
           numerized["Gender preferences of your roommate"],numerized["Say it's a Friday night, where would you most likely be?"],
           numerized["What's the first thing you do when you get out of bed?"],
           numerized["Do you have a partner?"],
           numerized["So, how often does your partner visit?"],numerized["What is your budget? (per person)"],numerized["How many roommates are you comfortable with?"]],axis=1)
#heomdat
#if heomdat.loc[heomdat['Gender preferences of your roommate'] == 0]
 # print heomdat.loc[heomdat['Gender'] == 0]
    heomdat1 = pd.DataFrame()
    if heomdat.iloc[-1,14] == 1:
        heomdat1 = heomdat.loc[heomdat['Gender'] == 1]
    elif heomdat.iloc[-1,14] == 0:
        heomdat1 = heomdat.loc[heomdat['Gender'] == 0]
    elif heomdat.iloc[-1,14] == 2:
        heomdat1 = heomdat
#print(heomdat1)

    heomdat1 = heomdat1.drop('Gender',axis=1)
    names1 = heomdat1['Full Name']
    Emailid = heomdat1['Email Address']


    heomdat1 = heomdat1.drop('Full Name',axis=1)
    heomdat1 = heomdat1.drop('Email Address',axis=1)
    heomdat1 = heomdat1.reset_index(drop=True)
    names1 = names1.reset_index(drop=True)
    Emailid = Emailid.reset_index(drop=True)

 
    categorical_ix = [0,17]
# Declare the HEOM with a correct NaN equivalent value
    heom_metric = HEOM(heomdat1, categorical_ix)
# Declare NearestNeighbor and link the metric
    neighbor = NearestNeighbors(metric = heom_metric.heom)
# Fit the model which uses the custom distance metric 
    neighbor.fit(heomdat1)
    nn= neighbor.fit(heomdat1)
#nn
    df = pd.DataFrame(heomdat1)
    df1=df.to_numpy()

# Return 5-Nearest Neighbors to the 1st instance (row 1)
    resultdf= []
#for i in range(0,len(df1)):
    result = neighbor.kneighbors(df1[-1].reshape(1, -1), n_neighbors = 5)
    resultdf.append(result[1])
#resultdf
    resultdf1 = np.delete(resultdf,0)
    res2 = pd.DataFrame(resultdf1)
  #%%  
    resultmatch= []

    resulta = neighbor.kneighbors(df1[-1].reshape(1, -1), n_neighbors = 5)
    resultmatch.append(resulta[0])
    
    resultformaxdist= []
    for i in range(0,len(df1)):
        resulta = neighbor.kneighbors(df1[i].reshape(1, -1), n_neighbors = 5)
        resultformaxdist.append(resulta[0])
    resformax=pd.DataFrame()
    resformax1=pd.DataFrame()

    resformax = np.delete(resultformaxdist,0)
    resi = pd.DataFrame(resformax)
    maxdist=max(resi[0])
    
    reco2= np.delete(resultmatch,0)

    reco2 = pd.DataFrame(reco2)
    
    
    origin= np.delete(resultmatch,-3)

    origin = pd.DataFrame(origin)
    
    origin = origin.iloc[0]
    
    match=[]
    for i in range(0,4):
        m = (1- ((reco2.iloc[i]-origin)/maxdist))*100
        match.append(m)
    matchdf =pd.DataFrame(columns=['Recommendations'])
    match = pd.DataFrame(match)
    m1 = pd.DataFrame([match.iloc[0]])
    m2 = pd.DataFrame([match.iloc[1]])
    m3 = pd.DataFrame([match.iloc[2]])
    m4 = pd.DataFrame([match.iloc[3]])
    matchdf =pd.concat([m1,m2,m3,m4],axis=0,ignore_index=True)
    recon1 = res2.iloc[0,0]
    recon2 = res2.iloc[1,0]
    recon3 = res2.iloc[2,0]
    recon4 = res2.iloc[3,0]
    


    id3 = pd.DataFrame([Emailid.iloc[recon1]], columns=['Recommendations'])
    id4 = pd.DataFrame([Emailid.iloc[recon2]], columns=['Recommendations'])
    id5 = pd.DataFrame([Emailid.iloc[recon3]], columns=['Recommendations'])
    id6 = pd.DataFrame([Emailid.iloc[recon4]], columns=['Recommendations'])
    dfid = pd.concat([id3, id4,id5,id6], ignore_index=True)
    dfid = pd.DataFrame(dfid)
    
    
    #%%

#res2=res2.drop(res2.index[0])
  
   
    

 #recon3
# recon4 = res2.iloc[1,0]
#res = [int(sub.split(',')[1]) for sub in resultdf]
#recon1

 
 #map index with name
 #res3 = names.iloc[recon1]
    res3 = pd.DataFrame([names1.iloc[recon1]], columns=['Recommendations'])
    res4 = pd.DataFrame([names1.iloc[recon2]], columns=['Recommendations'])
    res5 = pd.DataFrame([names1.iloc[recon3]], columns=['Recommendations'])
    res6 = pd.DataFrame([names1.iloc[recon4]], columns=['Recommendations'])

# res4 = names.iloc[recon2]
 #res5 = names.iloc[recon3]
 #res6 = names.iloc[recon4]
#res3 = pd.DataFrame(columns=[res3])
#res4  = pd.DataFrame(columns=[res4])
#res5 = pd.DataFrame(columns=[res5])
#res6 = pd.DataFrame(columns=[res6])
 
 #<p>Please fill out our form below to get great roommate recommendations. We use an efficent 
  # algorithm to bring out the best combinations. We ask different types of questions to understand
   #the behaviour, personality and preferences of the individual. </p>
#res3
    df_row_reindex = pd.concat([res3, res4,res5,res6], ignore_index=True)
    withmatch = pd.concat([df_row_reindex,matchdf,dfid],axis=1,ignore_index=True)
    withmatch.columns = ['Name', 'Percentage Match', 'Email ID']


#df_row_reindex

#text recom

    preferences = data.iloc[:,-2]
#Define a TF-IDF Vectorizer Object. Remove all english stopwords
    tfidf = TfidfVectorizer(stop_words='english')
    #Replace NaN with an empty string
    preferences = preferences.fillna('')

#Construct the required TF-IDF matrix by applying the fit_transform method on the overview feature
    tfidf_matrix = tfidf.fit_transform(preferences)
#Output the shape of tfidf_matrix
 #tfidf_matrix
 #tfidf_matrix.shape

    dense = tfidf_matrix.toarray()
 #dense
 #dense.shape
    den = pd.DataFrame(dense)
 #den


    categorical_ix = [1,20]
# Declare the HEOM with a correct NaN equivalent value
    heom_metric = HEOM(den, categorical_ix)

# Declare NearestNeighbor and link the metric
    textneighbor = NearestNeighbors(metric = heom_metric.heom)


# Fit the model which uses the custom distance metric 
    textneighbor.fit(den)
    nn= textneighbor.fit(den)
#nn

    dftext = pd.DataFrame(den)

    dfa1=dftext.to_numpy()

# Return 5-Nearest Neighbors to the 1st instance (row 1)
    resultdftext= []
#for i in range(0,len(df11)):
    result = textneighbor.kneighbors(dfa1[-1].reshape(1, -1), n_neighbors = 5)
    resultdftext.append(result[1])


#resultdftext


    resultdftext = np.delete(resultdftext,0)
    textres1 = pd.DataFrame(resultdftext)
#res2=res2.drop(res2.index[0])
  
    textrecon1 = textres1.iloc[0,0]
 #textrecon1
    textrecon2 = textres1.iloc[1,0]
    textrecon3 = textres1.iloc[2,0]
    textrecon4 = textres1.iloc[3,0]

 #recon3
# recon4 = res2.iloc[1,0]
#res = [int(sub.split(',')[1]) for sub in resultdf]
#textrecon3
 
 #map index with name
    textres3 = pd.DataFrame([names.iloc[textrecon1]], columns=['Recommendations'])
    textres4 = pd.DataFrame([names.iloc[textrecon2]], columns=['Recommendations'])
    textres5 = pd.DataFrame([names.iloc[textrecon3]], columns=['Recommendations'])
    textres6 = pd.DataFrame([names.iloc[textrecon4]], columns=['Recommendations'])
    df_row_reindextext = pd.concat([textres3, textres4,textres5,textres6], ignore_index=True)
#df_row_reindextext

    if numerized.iloc[-1,17] == 1 and numerized.iloc[textrecon1,7] == 1:
        blah = pd.DataFrame([names.iloc[textrecon1]], columns=['Name'])
    elif numerized.iloc[-1,17] == 2:
        blah = pd.DataFrame([names.iloc[textrecon1]], columns=['Name'])

    if numerized.iloc[-1,17] == 1 and numerized.iloc[textrecon2,7] == 1:
        blah = pd.DataFrame([names.iloc[textrecon2]], columns=['Name'])
    elif numerized.iloc[-1,17] == 2:
        blah = pd.DataFrame([names.iloc[textrecon2]], columns=['Name'])

    if numerized.iloc[-1,17] == 1 and numerized.iloc[textrecon3,7] == 1:
        blah = pd.DataFrame([names.iloc[textrecon3]], columns=['Name'])
    elif numerized.iloc[-1,17] == 2:
        blah = pd.DataFrame([names.iloc[textrecon3]], columns=['Name'])

    if numerized.iloc[-1,17] == 1 and numerized.iloc[textrecon4,7] == 1:
       blah = pd.DataFrame([names.iloc[textrecon4]], columns=['Name'])
    elif numerized.iloc[-1,17] == 2:
       blah = pd.DataFrame([names.iloc[textrecon4]], columns=['Name'])

#merging of dfs
#df_row_reindextext
#df_row_reindex
       #df_row_reindex
    finalrecom = pd.concat([withmatch[0:4], blah[0:len(blah)]], ignore_index=True)
    finalrecom["Email ID"].fillna("Contact Admin", inplace = True)
    
    finalrecom["Percentage Match"].fillna("Contact Admin", inplace = True)
    finalrecom =  pd.DataFrame(finalrecom.drop_duplicates("Name"))
 #finalrecom
    #return finalrecom.to_html(header="true", table_id="table")
    return render_template('output_recommend.html', tables=[finalrecom.to_html(classes='data')],titles=finalrecom.columns.values)



@app.route('/success1')
def success1():
    scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name('sheet.json', scope)
    client = gspread.authorize(creds)
    sheet = client.open('Roommate').sheet1
    Responses = sheet.get_all_records()
#print(Responses)
    data= pd.DataFrame(Responses)
#data = data.drop([35])
    names = data['Full Name']
    Gender = data['Gender preferences of your roommate']
    numerized=data
    numerized['Room preferences'] = np.where(data['Room preferences'].apply(lambda x: 'and' in x), 1, 0)
    numerized['Gender'] = np.where(data['Gender'].apply(lambda x: 'Male' in x), 0, 1)
    numerized['Occupation'] = np.where(data['Occupation'].apply(lambda x: 'Student' in x), 1, 0)
    numerized['Are you a Vegetarian?'] = np.where(data['Are you a Vegetarian?'].apply(lambda x: 'Yes' in x), 1, 0)
    numerized['Grocery shopping'] = np.where(data['Grocery shopping'].apply(lambda x: 'Would keep grocery separate' in x), 0, 1)
    numerized['Do you smoke? (and smoke up?)'] = np.where(data['Do you smoke? (and smoke up?)'].apply(lambda x: 'Yes' in x), 1, 0)
    numerized['Do you drink?'] = np.where(data['Do you drink?'].apply(lambda x: 'Yes' in x), 1, 0)
    numerized['Is it cool with you if your roommate organizes house parties?'] = np.where(data['Is it cool with you if your roommate organizes house parties?'].apply(lambda x: 'Yes' in x), 1, 0)
    numerized['Okay last question! When are your eyes first wide open in the morning?'] = np.where(data['Okay last question! When are your eyes first wide open in the morning?'].apply(lambda x: 'Early morning (anytime before 7 am easily)' in x), 0, 1)
    numerized['Be honest, are you a procrastinator? '] = np.where(data['Be honest, are you a procrastinator? '].apply(lambda x: 'Yes' in x), 1, 0)

#numerized["House Cleaning and Cooking"].value_counts()
    num1 = {"House Cleaning and Cooking": {"Maid for both": 0, "Maid for Cooking only": 1,"Maid for none": 2,"Would like to keep maid for housecleaning": 3}}
    numerized.replace(num1, inplace=True)

#numerized["So what are your work timings?"].value_counts()
    num2 = {"So what are your work timings?": {"Day shift": 0, "Varies": 1,"Night shift": 2}}
    numerized.replace(num2, inplace=True)

#numerized["Gender preferences of your roommate"].value_counts()
    num3 = {"Gender preferences of your roommate": {"Male": 0, "Female": 1, "Indifferent": 2}}
    numerized.replace(num3, inplace=True)

#numerized["Say it's a Friday night, where would you most likely be?"].value_counts()
    num4 = {"Say it's a Friday night, where would you most likely be?": {"play Football or hit the gym": 0, "The Club! (obviously!)": 0, "Read a book or watch a movie curled up  like a burrito (ah, sounds nice just reading it!)": 1,"Work doesn't allow for such luxuries":1}}
    numerized.replace(num4, inplace=True)

#numerized["What's the first thing you do when you get out of bed?"].value_counts()
    num5 = {"What's the first thing you do when you get out of bed?": {"Go back to sleep, until it's too late to be acceptably late (we've all been there)": 0, "Make the bed": 1, "Start planning my day ahead": 1, "Knock over stuff on the floor":2}}
    numerized.replace(num5, inplace=True)

#numerized["Do you have a partner?"].value_counts()
    num6 = {"Do you have a partner?": {"No": 0, "Prefer not to say": 0, "Yes": 1}}
    numerized.replace(num6, inplace=True)

#numerized["So, how often does your partner visit?"].value_counts()
    num7 = {"So, how often does your partner visit?": {"Na": 0,"": 0, "Never": 1,"1-2 a week" : 1,"Rarely": 1,"once in a month or two": 2, "well, almost all the time, because they're the best human being on the planet!": 2}}
    numerized.replace(num7, inplace=True)

#numerized["Is it cool if your roommate's partner visits often?"].value_counts()
    num8 = {"Is it cool if your roommate's partner visits often?": {"No": 0, "Yes": 1}}
    numerized.replace(num8, inplace=True)

#numerized["What is your budget? (per person)"].value_counts()
    num9 = {"What is your budget? (per person)": {"4000-6000": 0, "6000-8000": 1,"8000-10000":2,"10000-12000":3}}
    numerized.replace(num9, inplace=True)
#numerized
    h=pd.DataFrame(numerized['Room preferences'])
    heomdat=pd.concat([h,data['Full Name'],data['Email Address'],numerized['Gender'],numerized['Occupation'],
           numerized['Are you a Vegetarian?'],numerized['Grocery shopping'],
           numerized['Do you smoke? (and smoke up?)'],numerized['Do you drink?'],
           numerized['Is it cool with you if your roommate organizes house parties?'],
           numerized['Okay last question! When are your eyes first wide open in the morning?'],
           numerized['Be honest, are you a procrastinator? '],
           numerized["House Cleaning and Cooking"],
           numerized["So what are your work timings?"],
           numerized["Gender preferences of your roommate"],numerized["Say it's a Friday night, where would you most likely be?"],
           numerized["What's the first thing you do when you get out of bed?"],
           numerized["Do you have a partner?"],
           numerized["So, how often does your partner visit?"],numerized["What is your budget? (per person)"],numerized["How many roommates are you comfortable with?"]],axis=1)
#heomdat
#if heomdat.loc[heomdat['Gender preferences of your roommate'] == 0]
 # print heomdat.loc[heomdat['Gender'] == 0]
    heomdat1 = pd.DataFrame()
    if heomdat.iloc[-1,14] == 1:
        heomdat1 = heomdat.loc[heomdat['Gender'] == 1]
    elif heomdat.iloc[-1,14] == 0:
        heomdat1 = heomdat.loc[heomdat['Gender'] == 0]
    elif heomdat.iloc[-1,14] == 2:
        heomdat1 = heomdat
#print(heomdat1)

    heomdat1 = heomdat1.drop('Gender',axis=1)
    names1 = heomdat1['Full Name']
    Emailid = heomdat1['Email Address']


    heomdat1 = heomdat1.drop('Full Name',axis=1)
    heomdat1 = heomdat1.drop('Email Address',axis=1)
    heomdat1 = heomdat1.reset_index(drop=True)
    names1 = names1.reset_index(drop=True)
    Emailid = Emailid.reset_index(drop=True)

 
    categorical_ix = [0,17]
# Declare the HEOM with a correct NaN equivalent value
    heom_metric = HEOM(heomdat1, categorical_ix)
# Declare NearestNeighbor and link the metric
    neighbor = NearestNeighbors(metric = heom_metric.heom)
# Fit the model which uses the custom distance metric 
    neighbor.fit(heomdat1)
    nn= neighbor.fit(heomdat1)
#nn
    df = pd.DataFrame(heomdat1)
    df1=df.to_numpy()

# Return 5-Nearest Neighbors to the 1st instance (row 1)
    resultdf= []
#for i in range(0,len(df1)):
    result = neighbor.kneighbors(df1[-1].reshape(1, -1), n_neighbors = 5)
    resultdf.append(result[1])
#resultdf
    resultdf1 = np.delete(resultdf,0)
    res2 = pd.DataFrame(resultdf1)
  #%%  
    resultmatch= []

    resulta = neighbor.kneighbors(df1[-1].reshape(1, -1), n_neighbors = 5)
    resultmatch.append(resulta[0])
    
    resultformaxdist= []
    for i in range(0,len(df1)):
        resulta = neighbor.kneighbors(df1[i].reshape(1, -1), n_neighbors = 5)
        resultformaxdist.append(resulta[0])
    resformax=pd.DataFrame()
    resformax1=pd.DataFrame()

    resformax = np.delete(resultformaxdist,0)
    resi = pd.DataFrame(resformax)
    maxdist=max(resi[0])
    
    reco2= np.delete(resultmatch,0)

    reco2 = pd.DataFrame(reco2)
    
    
    origin= np.delete(resultmatch,-3)

    origin = pd.DataFrame(origin)
    
    origin = origin.iloc[0]
    
    match=[]
    for i in range(0,4):
        m = (1- ((reco2.iloc[i]-origin)/maxdist))*100
        match.append(m)
    matchdf =pd.DataFrame(columns=['Recommendations'])
    match = pd.DataFrame(match)
    m1 = pd.DataFrame([match.iloc[0]])
    m2 = pd.DataFrame([match.iloc[1]])
    m3 = pd.DataFrame([match.iloc[2]])
    m4 = pd.DataFrame([match.iloc[3]])
    matchdf =pd.concat([m1,m2,m3,m4],axis=0,ignore_index=True)
    recon1 = res2.iloc[0,0]
    recon2 = res2.iloc[1,0]
    recon3 = res2.iloc[2,0]
    recon4 = res2.iloc[3,0]
    


    id3 = pd.DataFrame([Emailid.iloc[recon1]], columns=['Recommendations'])
    id4 = pd.DataFrame([Emailid.iloc[recon2]], columns=['Recommendations'])
    id5 = pd.DataFrame([Emailid.iloc[recon3]], columns=['Recommendations'])
    id6 = pd.DataFrame([Emailid.iloc[recon4]], columns=['Recommendations'])
    dfid = pd.concat([id3, id4,id5,id6], ignore_index=True)
    dfid = pd.DataFrame(dfid)
    
    
    #%%

#res2=res2.drop(res2.index[0])
  
   
    

 #recon3
# recon4 = res2.iloc[1,0]
#res = [int(sub.split(',')[1]) for sub in resultdf]
#recon1

 
 #map index with name
 #res3 = names.iloc[recon1]
    res3 = pd.DataFrame([names1.iloc[recon1]], columns=['Recommendations'])
    res4 = pd.DataFrame([names1.iloc[recon2]], columns=['Recommendations'])
    res5 = pd.DataFrame([names1.iloc[recon3]], columns=['Recommendations'])
    res6 = pd.DataFrame([names1.iloc[recon4]], columns=['Recommendations'])

# res4 = names.iloc[recon2]
 #res5 = names.iloc[recon3]
 #res6 = names.iloc[recon4]
#res3 = pd.DataFrame(columns=[res3])
#res4  = pd.DataFrame(columns=[res4])
#res5 = pd.DataFrame(columns=[res5])
#res6 = pd.DataFrame(columns=[res6])
 
 #<p>Please fill out our form below to get great roommate recommendations. We use an efficent 
  # algorithm to bring out the best combinations. We ask different types of questions to understand
   #the behaviour, personality and preferences of the individual. </p>
#res3
    df_row_reindex = pd.concat([res3, res4,res5,res6], ignore_index=True)
    withmatch = pd.concat([df_row_reindex,matchdf,dfid],axis=1,ignore_index=True)
    withmatch.columns = ['Name', 'Percentage Match', 'Email ID']


#df_row_reindex

#text recom

    preferences = data.iloc[:,-2]
#Define a TF-IDF Vectorizer Object. Remove all english stopwords
    tfidf = TfidfVectorizer(stop_words='english')
    #Replace NaN with an empty string
    preferences = preferences.fillna('')

#Construct the required TF-IDF matrix by applying the fit_transform method on the overview feature
    tfidf_matrix = tfidf.fit_transform(preferences)
#Output the shape of tfidf_matrix
 #tfidf_matrix
 #tfidf_matrix.shape

    dense = tfidf_matrix.toarray()
 #dense
 #dense.shape
    den = pd.DataFrame(dense)
 #den


    categorical_ix = [1,20]
# Declare the HEOM with a correct NaN equivalent value
    heom_metric = HEOM(den, categorical_ix)

# Declare NearestNeighbor and link the metric
    textneighbor = NearestNeighbors(metric = heom_metric.heom)


# Fit the model which uses the custom distance metric 
    textneighbor.fit(den)
    nn= textneighbor.fit(den)
#nn

    dftext = pd.DataFrame(den)

    dfa1=dftext.to_numpy()

# Return 5-Nearest Neighbors to the 1st instance (row 1)
    resultdftext= []
#for i in range(0,len(df11)):
    result = textneighbor.kneighbors(dfa1[-1].reshape(1, -1), n_neighbors = 5)
    resultdftext.append(result[1])


#resultdftext


    resultdftext = np.delete(resultdftext,0)
    textres1 = pd.DataFrame(resultdftext)
#res2=res2.drop(res2.index[0])
  
    textrecon1 = textres1.iloc[0,0]
 #textrecon1
    textrecon2 = textres1.iloc[1,0]
    textrecon3 = textres1.iloc[2,0]
    textrecon4 = textres1.iloc[3,0]

 #recon3
# recon4 = res2.iloc[1,0]
#res = [int(sub.split(',')[1]) for sub in resultdf]
#textrecon3
 
 #map index with name
    textres3 = pd.DataFrame([names.iloc[textrecon1]], columns=['Recommendations'])
    textres4 = pd.DataFrame([names.iloc[textrecon2]], columns=['Recommendations'])
    textres5 = pd.DataFrame([names.iloc[textrecon3]], columns=['Recommendations'])
    textres6 = pd.DataFrame([names.iloc[textrecon4]], columns=['Recommendations'])
    df_row_reindextext = pd.concat([textres3, textres4,textres5,textres6], ignore_index=True)
#df_row_reindextext

    if numerized.iloc[-1,17] == 1 and numerized.iloc[textrecon1,7] == 1:
        blah = pd.DataFrame([names.iloc[textrecon1]], columns=['Name'])
    elif numerized.iloc[-1,17] == 2:
        blah = pd.DataFrame([names.iloc[textrecon1]], columns=['Name'])

    if numerized.iloc[-1,17] == 1 and numerized.iloc[textrecon2,7] == 1:
        blah = pd.DataFrame([names.iloc[textrecon2]], columns=['Name'])
    elif numerized.iloc[-1,17] == 2:
        blah = pd.DataFrame([names.iloc[textrecon2]], columns=['Name'])

    if numerized.iloc[-1,17] == 1 and numerized.iloc[textrecon3,7] == 1:
        blah = pd.DataFrame([names.iloc[textrecon3]], columns=['Name'])
    elif numerized.iloc[-1,17] == 2:
        blah = pd.DataFrame([names.iloc[textrecon3]], columns=['Name'])

    if numerized.iloc[-1,17] == 1 and numerized.iloc[textrecon4,7] == 1:
       blah = pd.DataFrame([names.iloc[textrecon4]], columns=['Name'])
    elif numerized.iloc[-1,17] == 2:
       blah = pd.DataFrame([names.iloc[textrecon4]], columns=['Name'])

#merging of dfs
#df_row_reindextext
#df_row_reindex
       #df_row_reindex
    finalrecom = pd.concat([withmatch[0:4], blah[0:len(blah)]], ignore_index=True)
    finalrecom["Email ID"].fillna("Contact Admin", inplace = True)
    
    finalrecom["Percentage Match"].fillna("Contact Admin", inplace = True)
    finalrecom =  pd.DataFrame(finalrecom.drop_duplicates("Name"))

    from tabulate import tabulate

    import smtplib
 
    def sendemail(from_addr, to_addr_list, cc_addr_list,
              subject, message,
              login, password,
              smtpserver='smtp.gmail.com:587'):
        header  = 'From: %s\n' % from_addr
        header += 'To: %s\n' % ','.join(to_addr_list)
        header += 'Cc: %s\n' % ','.join(cc_addr_list)
        header += 'Subject: %s\n\n' % subject
        message = header + message
 
        server = smtplib.SMTP(smtpserver)
        server.starttls()
        server.login(login,password)
        problems = server.sendmail(from_addr, to_addr_list, message)
        server.quit()
        return problems
    emaillist = pd.DataFrame()
    emaillist = pd.DataFrame(["pgdm18krishnapriya@mse.ac.in", "pgdm18ashwati@mse.ac.in","pgdm18komal@mse.ac.in","pgdm18arushi@mse.ac.in"])
    namelist =pd.DataFrame(["KP","Ashwati","Komal","Arushi"])
    matchlist = pd.DataFrame([])
    message = withmatch
    me =["Hey there!Thank you for using Roomie Reco to find your perfect roommate!",
         "According to our sophisticated algorithm, here are your potential roommates:",
         '\n',"With a ", message.iloc[0,1], " percent match is ", message.iloc[0,0]," their Email ID is: ",message.iloc[0,2],'\n'
         '\n',"With a ", message.iloc[1,1], " percent match is ", message.iloc[1,0]," their Email ID is: ",message.iloc[1,2],'\n'
         '\n',"With a ", message.iloc[2,1], " percent match is ", message.iloc[2,0]," their Email ID is: ",message.iloc[2,2],'\n'
         '\n',"With a ", message.iloc[3,1], " percent match is ", message.iloc[3,0]," their Email ID is: ",message.iloc[3,2]]
#%%
    #%%
    my_lst_str = ''.join(map(str, me))
#%%
    sendemail(from_addr    = 'roomiereco@gmail.com', 
          to_addr_list = numerized.iloc[-1,30],
          cc_addr_list = ["pgdm18krishnapriya@mse.ac.in", "pgdm18ashwati@mse.ac.in","pgdm18komal@mse.ac.in"], 
          subject      = 'Your Recommended Roommate!', 
          message      =   my_lst_str,
          login        = 'roomiereco', 
          password     = 'datascience123')
    textdata=pd.DataFrame(["Please check your mail, awesome recommendations on your way"])
 #finalrecom
    return render_template('outputt1.html')

    
  
    


@app.route('/login',methods = ['POST', 'GET'])
def login():
    print("ABCSSSSSSSSSSSSSSS")
    user = request.form['art1']
    print(user)
    if(user == "See recommendations here"):
        return redirect(url_for('success'))
    else:
        return redirect(url_for('success1'))

if __name__ == '__main__':
   app.run()

