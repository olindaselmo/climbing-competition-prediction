import pandas as pd

athletes = pd.read_csv("data/athlete_information.csv")
results = pd.read_csv("data/athlete_results.csv")

# un site web m'a conseillé de convertir les dates qui arrivent depuis le fichier .csv
# sous forme de texte (chaînes de caractères), en vraies dates utilisables par Python
athletes['birthday'] = pd.to_datetime(athletes['birthday'], errors='coerce')
results['date'] = pd.to_datetime(results['date'], errors='coerce')

# Je filtre les résultats des 10 dernières années 
# car c'est la période pendant laquelle les grimpeurs actuels ont pu concour
anneelimite = pd.Timestamp.today().year - 10
resultatrecent = results[results['season'] >= anneelimite]

# Je garde uniquement les athlètes ayant concouru dans ces 10 dernières années
athletesrecent = athletes[athletes['athlete_id'].isin(resultatrecent['athlete_id'])]

# Je calcule l'âge de l'athlète par rapport à sa date de naissance 
today = pd.Timestamp.today()
athletesrecent['computed_age'] = (today - athletesrecent['birthday']).dt.days // 365

# Je cherche l'athlète le plus vieux (dans mon échantillon précédent)
# afin de garder tous les athlètes plus jeunes que celui-ci
plusvieux_athlete = athletesrecent.loc[athletesrecent['computed_age'].idxmax()]
agev = plusvieux_athlete['computed_age']

print("L’athlète le plus âgé à avoir concouru ces 10 dernières années :")
print(f"{plusvieux_athlete['firstname']} {plusvieux_athlete['lastname']} ({agev} ans)")

# Je vois que le plus âgé (Hans Ewald GRILL, 74 ans) est en paraclimpique.
# Je décide maintenant :
#  pour les disciplines paralympiques : garder jusqu’à 74 ans
#  pour les autres disciplines (lead, boulder, speed) : garder uniquement les moins de 45 ans

# Je récupère les athlètes avec leur discipline principale 
# en joignant les résultats 
filtrediscipline = results.merge(athletes, on='athlete_id', how='left')

# Je définis une condition pour séparer les paraclimpiques et les autres
is_para = filtrediscipline['paraclimbing_sport_class'].notna() & (filtrediscipline['paraclimbing_sport_class'].str.strip() != "")
filtrediscipline['keep'] = filtrediscipline.apply(
    lambda row: (
        row['age'] <= agev if is_para.loc[row.name]
        else row['age'] <= 45
    ),
    axis=1
)

# Je garde seulement les athlètes à conserver 
athletes_nettoyés = filtrediscipline[filtrediscipline['keep']].drop_duplicates(subset='athlete_id')

#j'obtiens : Nombre d'athlètes avant nettoyage : 16258
#            Nombre d'athlètes après nettoyage : 6611

# je rajoute une condition pour etre sure que les athlètes font toujours de la compétition: je 
#regarde si les athlètes ont fait une compétiton durant les 5 dernières années, sinon je les supprime
limite5ans = 2019 #car les données s'arrêtes en 2024. donc 2024-5=2019
athletes_compet_recent5 = results[results['season'] >= limite5ans]['athlete_id'].unique()
athletes_nettoyés2 = athletes_nettoyés[athletes_nettoyés['athlete_id'].isin(athletes_compet_recent5)]

print(f"\nNombre d'athlètes avant nettoyage : {len(athletes)}")
print(f"Nombre d'athlètes après nettoyage : {len(athletes_nettoyés2)}")

#j'obtiens : Nombre d'athlètes avant nettoyage : 16258
#            Nombre d'athlètes après nettoyage : 4617

# je sauvegarde dans un nouveau fichier les nouvelles données
athletes_nettoyés2.to_csv("data/cleaned_athlete_information.csv", index=False)
print("\n Fichier 'cleaned_athlete_information.csv' enregistré.")


