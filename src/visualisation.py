import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

top10_hommes = pd.read_csv("top10_hommes_XGB_2026.csv")
top10_femmes = pd.read_csv("top10_femmes_XGB_2026.csv")

#Normalisation des disciplines
top10_hommes['discipline'] = top10_hommes['discipline'].str.strip().str.lower()
top10_femmes['discipline'] = top10_femmes['discipline'].str.strip().str.lower()

# Concaténer prénom et nom
top10_hommes['athlete_name'] = top10_hommes['firstname'] + " " + top10_hommes['lastname']
top10_femmes['athlete_name'] = top10_femmes['firstname'] + " " + top10_femmes['lastname']

palette = {
    'boulder': 'blue',
    'lead': 'green',
    'boulder&lead': 'orange',
    'speed': 'red'
}


def plot_top10(df, genre):
    df_sorted = df.sort_values(by=['discipline', 'prob_top10'], ascending=[True, False])
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=df_sorted,
        x='prob_top10',
        y='athlete_name',
        hue='discipline',
        dodge=False,
        palette=palette
    )
    plt.title(f"Top 10 {genre} par discipline (probabilité de top10)")
    plt.xlabel("Probabilité de finir dans le top10")
    plt.ylabel("Athlète")
    plt.xlim(0, 1)
    plt.legend(title='Discipline')
    plt.tight_layout()
    plt.show()


plot_top10(top10_hommes, "Hommes")
plot_top10(top10_femmes, "Femmes")
