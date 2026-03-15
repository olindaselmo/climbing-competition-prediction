import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#pareil que autres codes
athletes = pd.read_csv("../data/cleaned_athlete_information.csv")

for col in ['age', 'height', 'arm_span', 'season', 'rank']:athletes[col] = pd.to_numeric(athletes[col], errors='coerce')
athletes = athletes.applymap(lambda x: x.strip() if isinstance(x, str) else x)
vrai_resultats = pd.read_csv("../data/vrairesultWC.csv")

for col in ['firstname', 'lastname', 'country', 'discipline']:
    athletes[col] = athletes[col].astype(str).str.strip().str.lower()
    vrai_resultats[col] = vrai_resultats[col].astype(str).str.strip().str.lower()

athletes['athlete_key'] = athletes['firstname'] + '_' + athletes['lastname'] + '_' + athletes['country'] + '_' + athletes['discipline']
vrai_resultats['athlete_key'] = vrai_resultats['firstname'] + '_' + vrai_resultats['lastname'] + '_' + vrai_resultats['country'] + '_' + vrai_resultats['discipline']

top2025_keys = set(vrai_resultats['athlete_key'])

boulderlead_athletes = athletes[(athletes['discipline'] == 'boulder&lead') & (athletes['rank'] <= 10)]
boulderlead_keys = set(boulderlead_athletes['firstname'] + '_' + boulderlead_athletes['lastname'] + '_' + boulderlead_athletes['country'])

athletes['bonus2025'] = athletes['athlete_key'].isin(top2025_keys).astype(int)
athletes['bonus_combined'] = (athletes['firstname'] + '_' + athletes['lastname'] + '_' + athletes['country']).isin(boulderlead_keys).astype(int)


# Supprimer les boulder&lead pour l'entraînement
athletes_train = athletes[athletes['discipline'] != 'boulder&lead'].copy()

# Filtrer les athlètes récents
recent_athletes = athletes_train[athletes_train['season'] >= 2019].copy()
recent_athletes['top10'] = recent_athletes['rank'] <= 10

#va
X_num = recent_athletes[['age', 'height', 'arm_span', 'bonus2025', 'bonus_combined']].fillna(0)
X_cat = recent_athletes[['gender', 'discipline', 'paraclimbing_sport_class']].fillna('None')

# OneHotEncoding pour les catégories
encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
X_cat_encoded = pd.DataFrame(encoder.fit_transform(X_cat), columns=encoder.get_feature_names_out(X_cat.columns))

# Combiner numériques + catégorielles
X = pd.concat([X_num.reset_index(drop=True), X_cat_encoded.reset_index(drop=True)], axis=1)
y = recent_athletes['top10'].astype(int)

# Standardisation (important pour modèles linéaires)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#Lofistique regression
clf = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
clf.fit(X_train, y_train)

#Prédiction
y_pred_proba = clf.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)  # seuil 0.5

#Évaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy du modèle : {accuracy:.3f}")
print("\nRapport de classification :")
print(classification_report(y_test, y_pred))
print("Matrice de confusion :")
print(confusion_matrix(y_test, y_pred))


X_num_all = athletes_train[['age', 'height', 'arm_span', 'bonus2025', 'bonus_combined']].fillna(0)
X_cat_all = athletes_train[['gender', 'discipline', 'paraclimbing_sport_class']].fillna('None')
X_cat_all_encoded = pd.DataFrame(encoder.transform(X_cat_all), columns=encoder.get_feature_names_out(X_cat_all.columns))
X_all = pd.concat([X_num_all.reset_index(drop=True), X_cat_all_encoded.reset_index(drop=True)], axis=1)
X_all_scaled = scaler.transform(X_all)

athletes_train['prob_top10'] = clf.predict_proba(X_all_scaled)[:, 1]

#bonus
athletes_train.loc[athletes_train['bonus2025'] == 1, 'prob_top10'] *= 1.15
athletes_train.loc[athletes_train['bonus_combined'] == 1, 'prob_top10'] *= 1.05
athletes_train['prob_top10'] = athletes_train['prob_top10'].clip(0, 1)

def get_top10(df):
    df_agg = df.groupby(['discipline', 'firstname', 'lastname', 'country']).agg({
        'prob_top10': 'mean',
        'age': 'mean',
        'height': 'mean',
        'arm_span': 'mean',
        'gender': 'first'
    }).reset_index()
    top10 = df_agg.sort_values(['discipline', 'prob_top10'], ascending=[True, False]).groupby('discipline').head(10)
    return top10

athletes_male = athletes_train[athletes_train['gender'] == 'male']
athletes_female = athletes_train[athletes_train['gender'] == 'female']

top10_hommes = get_top10(athletes_male)
top10_femmes = get_top10(athletes_female)

top10_hommes.to_csv("top10_hommes_logreg.csv", index=False)
top10_femmes.to_csv("top10_femmes_logreg.csv", index=False)

print("\nTop 10 par discipline et par genre générés avec Logistic Regression")
