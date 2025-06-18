# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 18:45:15 2024

@author: Util
"""
import pandas as pd
import streamlit as st
import math
from scipy.spatial import cKDTree
import numpy as np
from pyproj import Transformer
import matplotlib.pyplot as plt
import osmnx as ox
from geopy.exc import GeocoderTimedOut 
from scipy.spatial.distance import cdist
import geopandas as gpd
import pickle
from scipy import stats

import requests
import io
#import pyodbc
 
from datetime import datetime

import os



API_BAN_URL = 'https://api-adresse.data.gouv.fr/search/csv/'

 
passage_postal_insee = pd.read_csv("passage_postal_insee.csv",sep=";")  
passage_postal_insee.columns= ['code_insee', 'code_postal']
passage_postal_insee['code_insee'] =  passage_postal_insee['code_insee'].apply(lambda x: ("0"+str(x))[-5:])
passage_postal_insee['code_postal'] =  passage_postal_insee['code_postal'].apply(lambda x: ("0"+str(x))[-5:])
 
 
 

# Charger les coefficients à partir d'un fichier CSV
df_coefficients = pd.read_excel('coefficients_modeles.xlsx'  )
#contours_departements  = gpd.read_file(r'C:\Users\david\DVF\contour-des-departements.geojson')

df_coefficients['C'] = 0

for ii in list(df_coefficients):
    if ii not in ['variable','modalité','type']:
        df_coefficients[ii] = df_coefficients[ii] + df_coefficients['main']
        
df_coefficients = df_coefficients.drop(columns=['main'])

df_geographique= pd.DataFrame(columns=["x","y","pred_exp","zonage"])
for ii in ["A","C","Abis","B1","B2"]:
    df_geographique_inter=pd.read_parquet("base_geo_predi_corrig_"+str(ii)+".parquet", engine='pyarrow')
    df_geographique_inter['zonage']=str(ii)
    df_geographique = pd.concat([df_geographique,df_geographique_inter])

df_geographique = df_geographique.rename(columns={"pred_exp":'constante_exp'})
df_geographique = df_geographique.reset_index(drop=True)

df_geographique_unique = df_geographique.drop_duplicates(subset=['x','y']).reset_index(drop=True)

# Construire un arbre k-d pour rechercher les points les plus proches
tree = cKDTree(df_geographique_unique[['x', 'y']].values)

transformer = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)


# Fonction pour classer la surface bâtie
def classer_surface_batie(surface):
    quantile=  [0,30,60,80,110,150]
    if surface < quantile[1]:
        return "moins de " +str(int(quantile[1]))+" m²"
    elif quantile[1] <= surface <= quantile[2]:
        return "entre " +str(int(quantile[1]))+" et "+str(int(quantile[2])) +" m²"
    elif quantile[2] <= surface <= quantile[3]:
        return "entre " +str(int(quantile[2]))+" et "+str(int(quantile[3])) +" m²"
    elif quantile[3] <= surface <= quantile[4]:
        return "entre " +str(int(quantile[3]))+" et "+str(int(quantile[4])) +" m²"
    elif quantile[4] <= surface <= quantile[5]:
        return "entre " +str(int(quantile[4]))+" et "+str(int(quantile[5])) +" m²"
    else:
        return "plus de  " +str(int(quantile[5])) +" m²"

# Fonction pour calculer la constante géographique pondérée
def calculer_constante_geographique_et_region(x, y):
    distances, indices = tree.query([x, y], k=3)
    
    distances_max = distances[0]
    
    zonage_proche = df_geographique_unique.iloc[indices[0]]['zonage']
    
    constantes_proches = df_geographique_unique.iloc[indices]['constante_exp'].values
    poids = [math.exp(-xx/(distances_max))  for xx in list(distances) ]
    constante_ponderee = np.average(constantes_proches, weights=poids)
    return constante_ponderee, zonage_proche 

# Fonction pour calculer l'estimation
def calculer_estimation(region, variables_selectionnees):
    estimation = 0
    for variable, modalite in variables_selectionnees.items():
        coeff = df_coefficients.loc[
            (df_coefficients['variable'] == variable) & 
            (df_coefficients['modalité'] == modalite), 
            region
        ]
        if not coeff.empty:
            estimation += coeff.values[0]
    return estimation
 
 

def geocode_address_ban(adress):
    corr = 0
    location = (None, None)
    res =None, None, None
    
    if adress[-5:].isdigit() and adress[-6]==" " :
        code_postal = adress[-5:]
        numero_et_nomvoie = adress[:-6]
        code_insee = passage_postal_insee['code_insee'][passage_postal_insee['code_postal']==code_postal]
        if len(code_insee)>0:
            code_insee = code_insee.reset_index(drop=True).iloc[0]
        else:
            code_insee=""
        
        try:
            data_frame = pd.DataFrame([[code_insee,code_postal,numero_et_nomvoie]],columns=['code_insee', 'code_postal', 'numero_et_nomvoie'])            
            fichier_csv = data_frame.to_csv(index=False, sep=';')
            param_fichier = {                "data": fichier_csv            }
            param_appel_API = {"columns": 'code_insee',"postcode": 'code_postal',  "columns": 'numero_et_nomvoie',    }
                
            response = requests.post(API_BAN_URL,  files=param_fichier, data=param_appel_API)
            if response.status_code == 200:
                response = response.content
                reponse = pd.read_csv(io.StringIO(response.decode('utf-8')),sep = ';', dtype=object)
                if float(reponse['result_score'].iloc[0])>0.3:
                    location = (float(reponse['latitude'].iloc[0]),float(reponse['longitude'].iloc[0]))
                    corr = 1
                    res = location[1], location[0]   ,corr 

        except:
            corr = 0
              
    if corr==0: 
        try:        
                location = ox.geocode(address)
                res =  location[1],location[0]   ,corr 
        except:
             res =  None,None   ,corr 
    
    return   res   
  


st.title("Estimation pour hypothèque d'une opération")



# Entrée de l'adresse
adresse = st.text_input("Adresse = num et nom de voie + code postale", "33 rue des Lilas 75019")

longitude, latitude, =   2.396482,48.879845


# Géocodage si l'adresse est remplie
if len(adresse)>0:
    longitude, latitude,  corr = geocode_address_ban(adresse)  # Géocode et remplace les coordonnées
    
    
    if corr ==1:
        st.write(f"Géocodage de l'adresse {adresse} avec la ban")
        
    else:
        st.write(f"Géocodage de l'adresse {adresse} sans la ban")
    
    if latitude is None or longitude is None:
        st.error("Coordonnées GPS non valides pour l'adresse saisie.")
    

# Afficher les champs de latitude et de longitude, qui sont maintenant modifiés si l'adresse est saisie
latitude = st.number_input("Latitude (GPS)", value=latitude, format="%.6f")
longitude = st.number_input("Longitude (GPS)", value=longitude, format="%.6f")
surface_batie = st.number_input("Surface totale du bâtis (en m²)", min_value=10, step=1) 
nombre_logement = st.number_input("Nombre de logements dans l'opération", min_value=1, step=1) 


# Vérification si les coordonnées GPS sont valides
if latitude is not None and longitude is not None:
    # Conversion des coordonnées GPS en coordonnées Lambert 93
    x_lambert, y_lambert = transformer.transform(longitude, latitude)
    #x, y = transformer.transform(2.413867, 48.880048)
    # Générer des menus déroulants pour chaque variable
    variables_selectionnees = {}
variables = df_coefficients['variable'].unique()

dico={}
   
for variable in variables:
    if variable != "surface du bâti":
       
        df = df_coefficients[df_coefficients['variable']==variable].reset_index(drop=True)
        
        index_defaut = 0
        if variable == "info vendeur":
            id_ =  list(df[df['modalité']=="PERSONNE PHYSIQUE"].index)
            if len(id_)>0:
                index_defaut = id_[0]
        if variable == "nature de la mutation":
            id_ =  list(df[df['modalité']=="vente"].index)
            if len(id_)>0:
                index_defaut = id_[0]
        if variable == "maison":
            id_ =  list(df[df['modalité']=="appartement"].index)
            if len(id_)>0:
                index_defaut = id_[0]
        if variable == "occupation du logement":
            id_ =  list(df[df['modalité']=="OCCUPATION PAR UN LOCATAIRE"].index)
            if len(id_)>0:
                index_defaut = id_[0]
        if variable == "periode de construction":
            id_ =  list(df[df['modalité']=="après 2017"].index)
            if len(id_)>0:
                index_defaut = id_[0]
        if variable == "surface du local":
            id_ =  list(df[df['modalité']=="Aucun"].index)
            if len(id_)>0:
                index_defaut = id_[0]
        if variable == 'nb pièces':
            id_ =  list(df[df['modalité']=="T2"].index)
            if len(id_)>0:
                index_defaut = id_[0]
        if variable == "Nombre d'étage de l'immeuble / maison":
            id_ =  list(df[df['modalité']=="-"].index)
            if len(id_)>0:
                index_defaut = id_[0]
         
        if variable == 'vente en bloc':
            id_ =  list(df[df['modalité']=='Non'].index)
            if len(id_)>0:
                index_defaut = id_[0]    
        if variable == "Nombre de logements (si vente en bloc)":
            id_ =  list(df[df['modalité']=='Pas de vente en bloc'].index)
            if len(id_)>0:
                index_defaut = id_[0]       
        if variable == "surface du terrain":
            id_ =  list(df[df['modalité']=="Pas de terrain"].index)
            if len(id_)>0:
                index_defaut = id_[0]       
        if variable == "garage":
            id_ =  list(df[df['modalité']=="Non"].index)
            if len(id_)>0:
                index_defaut = id_[0]               
        modalites = df_coefficients[df_coefficients['variable'] == variable]['modalité'].unique()
        dico[variable]=modalites
    
for variable in variables:
    if variable != "surface du bâti":        
        selection = st.selectbox(f"Choisissez une modalité pour {variable}", dico[variable],index=index_defaut)
        


# Bouton pour calculer l'estimation
if  st.button("Estimer"):
        # Calcul de la constante géographique et de la région en fonction des coordonnées Lambert
        modalite_surface = classer_surface_batie(surface_batie/nombre_logement)
        variables_selectionnees["surface du bâti"] = modalite_surface
        for variable in variables:
            if variable != "surface du bâti":        
                selection = st.selectbox(f"Choisissez une modalité pour {variable}", dico[variable],index=index_defaut)
                variables_selectionnees[variable] = selection
        
 
        constante_geographique,zonage  = calculer_constante_geographique_et_region(x_lambert, y_lambert)
        constante_geographique = round(constante_geographique, 0)
        #st.write(f"Departement déterminée automatiquement : {dep}")
        st.write(f"Zonage déterminée automatiquement : {zonage}")
        
        
        
        # Calcul de l'effet multiplicatif
        estimation_multiplicative = calculer_estimation(zonage, variables_selectionnees)
        effet_multiplicatif = round(math.exp(estimation_multiplicative), 2)
        
        # Résultat final
        estimation_finale = int(round(constante_geographique * effet_multiplicatif, 0))
        estimation_finale = int(estimation_finale * 1.0057) #passage du T4 2024 AU T1 2025
        
        # Afficher les résultats
        st.write(f"Effet multiplicatif : {effet_multiplicatif}")
        st.write(f"Constante géographique : {constante_geographique}")
        st.write(f"Résultat final (constante * effet multiplicatif) : {estimation_finale} euros par m²")
        
        
        #(ecart_95,ecart_90,residual_std_error) = recup_ligne_proxi(x_lambert, y_lambert,train_creuse.copy(),all_feature_names ,estimation_finale)
        
        #EC95_1 = int(100 * round(estimation_finale*math.exp(ecart_95)* surface_batie/100,0))
        #EC95_2 = int(100 *round(estimation_finale*math.exp(-ecart_95)* surface_batie/100,0))
        #EC90_1 = int(100 *round(estimation_finale*math.exp(ecart_90)* surface_batie/100,0))
        #EC90_2 = int(100 *round(estimation_finale*math.exp(-ecart_90)* surface_batie/100,0))
        
        #st.write(f"Intervalle de confiance à 95% : [ {EC95_2}  euros , {EC95_1} euros ]")
        #st.write(f"Intervalle de confiance à 90% : [ {EC90_2}  euros , {EC90_1} euros ]")
        
        
        # Calcul du prix total en fonction de la surface bâtie
        prix_total = int(100 * round(estimation_finale * surface_batie/100,0))
        st.write(f"Prix total du bien : {round(prix_total, 1)} €")
        
        
        # Affichage du graphique avec les points à proximité
       
        
        x_min, x_max = x_lambert - 5000, x_lambert + 5000
        y_min, y_max = y_lambert - 5000, y_lambert + 5000
        
        df_proximite = df_geographique[(df_geographique['x'] >= x_min) & (df_geographique['x'] <= x_max) & 
                 (df_geographique['y'] >= y_min) & (df_geographique['y'] <= y_max)]
        
        plt.figure(figsize=(10, 10))
      
        sc = plt.scatter(
            df_proximite['x'], df_proximite['y'], 
            c=df_proximite['constante_exp'], cmap='inferno', s=20, alpha=0.6,
            vmin=df_proximite['constante_exp'].quantile(0.002),
            vmax=df_proximite['constante_exp'].quantile(0.998)
        )
        
        plt.colorbar(sc, label='Valeurs géographiques')
        plt.scatter(x_lambert, y_lambert, c='black', s=250, marker='x', label="Point utilisateur", linewidth=3)
        plt.xlabel('Coordonnées Lambert 93 - X')
        plt.ylabel('Coordonnées Lambert 93 - Y')
        plt.title("Constantes géographiques autour du point utilisateur")
        plt.legend()
        st.pyplot(plt)
else:
    st.error("Veuillez saisir une adresse ou des coordonnées GPS valides.")

