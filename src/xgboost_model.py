import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb  
#meme chose que les autres modèles:
athletes = pd.read_csv("../data/cleaned_athlete_information.csv")

for col in ['age', 'height', 'arm_span', 'season', 'rank']:athletes[col] = pd.to_numeric(athletes[col], errors='coerce')

athletes = athletes.applymap(lambda x: x.strip() if isinstance(x, str) else x)
vrai_resultats = pd.read_csv("../data/vrairesultWC.csv")


for col in ['firstname', 'lastname', 'country', 'discipline']:
    vrai_resultats[col] = vrai_resultats[col].astype(str).str.strip().str.lower()
    athletes[col] = athletes[col].astype(str).str.strip().str.lower()


athletes['athlete_key'] = (
    athletes['firstname'] + '_' + athletes['lastname'] + '_' +
    athletes['country'] + '_' + athletes['discipline']
)
vrai_resultats['athlete_key'] = (
    vrai_resultats['firstname'] + '_' + vrai_resultats['lastname'] + '_' +
    vrai_resultats['country'] + '_' + vrai_resultats['discipline']
)

# Top 10 2025 WCpour bonus
top2025_keys = set(vrai_resultats['athlete_key'])

# Bonus pour anciens "boulder&lead"
boulderlead_athletes = athletes[
    (athletes['discipline'] == 'boulder&lead') & (athletes['rank'] <= 10)
]
boulderlead_keys = set(
    boulderlead_athletes['firstname'] + '_' +
    boulderlead_athletes['lastname'] + '_' +
    boulderlead_athletes['country']
)

# Ajouter les colonnes bonus au dataset complet
athletes['bonus2025'] = athletes['athlete_key'].isin(top2025_keys).astype(int)
athletes['bonus_combined'] = (
    (athletes['firstname'] + '_' + athletes['lastname'] + '_' + athletes['country']).isin(boulderlead_keys)
).astype(int)

# Supprimer la discipline boulder&lead pour l'entraînement
athletes_train = athletes[athletes['discipline'] != 'boulder&lead'].copy()

# Filtrer les athlètes récents
recent_athletes = athletes_train[athletes_train['season'] >= 2019].copy()
recent_athletes['top10'] = recent_athletes['rank'] <= 10

#va
X_num = recent_athletes[['age', 'height', 'arm_span', 'bonus2025', 'bonus_combined']].fillna(0)
X_cat = recent_athletes[['gender', 'discipline', 'paraclimbing_sport_class']].fillna('None')


encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
X_cat_encoded = pd.DataFrame(
    encoder.fit_transform(X_cat),
    columns=encoder.get_feature_names_out(X_cat.columns)
)

X = pd.concat([X_num.reset_index(drop=True), X_cat_encoded.reset_index(drop=True)], axis=1)
y = recent_athletes['top10'].astype(int)

#separation trrain test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#entrainement XGBOOST
clf = xgb.XGBClassifier(
    n_estimators=300,        # plus d'arbres pour la stabilité
    max_depth=6,             # profondeur des arbres
    learning_rate=0.1,       # taux d'apprentissage
    scale_pos_weight=3,      # pondérer la classe top10
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

clf.fit(X_train, y_train)

#eval modèle:
y_pred_proba = clf.predict_proba(X_test)[:, 1]
# On peut tester un seuil <0.5 pour détecter plus de top10
y_pred = (y_pred_proba >= 0.4).astype(int)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy du modèle : {round(accuracy, 3)}")
print("\nRapport de classification :")
print(classification_report(y_test, y_pred))
print("Matrice de confusion :")
print(confusion_matrix(y_test, y_pred))

#données finales
athletes_predict = athletes_train.copy()
X_num_all = athletes_predict[['age', 'height', 'arm_span', 'bonus2025', 'bonus_combined']].fillna(0)
X_cat_all = athletes_predict[['gender', 'discipline', 'paraclimbing_sport_class']].fillna('None')
X_cat_all_encoded = pd.DataFrame(
    encoder.transform(X_cat_all),
    columns=encoder.get_feature_names_out(X_cat_all.columns)
)
X_all = pd.concat([X_num_all.reset_index(drop=True), X_cat_all_encoded.reset_index(drop=True)], axis=1)

#prédictions des probas
athletes_predict['prob_top10'] = clf.predict_proba(X_all)[:, 1]

#bonus
athletes_predict.loc[athletes_predict['bonus2025'] == 1, 'prob_top10'] *= 1.15
athletes_predict.loc[athletes_predict['bonus_combined'] == 1, 'prob_top10'] *= 1.05
athletes_predict['prob_top10'] = athletes_predict['prob_top10'].clip(0, 1)


athletes_male = athletes_predict[athletes_predict['gender'] == 'male']
athletes_female = athletes_predict[athletes_predict['gender'] == 'female']


def get_top10(df):
    df_agg = (
        df.groupby(['discipline', 'firstname', 'lastname', 'country'])
          .agg({
              'prob_top10': 'mean',
              'age': 'mean',
              'height': 'mean',
              'arm_span': 'mean',
              'gender': 'first',
          })
          .reset_index()
    )
    top10 = (
        df_agg.sort_values(by=['discipline', 'prob_top10'], ascending=[True, False])
              .groupby('discipline')
              .head(10)
    )
    return top10


top10_hommes = get_top10(athletes_male)
top10_femmes = get_top10(athletes_female)

top10_hommes.to_csv("top10_hommes_XGB_2026.csv", index=False)
top10_femmes.to_csv("top10_femmes_XGB_2026.csv", index=False)

print("Top 10 pour 2026 ")
print(" - top10_hommes_XGB_2026.csv")
print(" - top10_femmes_XGB_2026.csv")
