import matplotlib.pyplot as plt
import numpy as np

def lire_fichier(chemin):
    with open(chemin, 'r') as fichier:
        lignes = fichier.readlines()
        donnees = []
        for ligne in lignes:
            ligne = ligne.strip().replace('[', '').replace(']', '')  # Supprimer les crochets
            valeurs = [float(val) for val in ligne.split(',')]
            donnees.append(np.array(valeurs))
        return donnees

def traiter_donnees(donnees):
    if len(donnees) < 2:
        print("Pas assez de données pour calculer la moyenne et l'écart-type.")
        return None, None, None
    
    x = donnees[0]
    y = np.array(donnees[1:])
    
    if y.size == 0:
        print("Les données ne sont pas au bon format.")
        return None, None, None
    
    y_moyenne = np.mean(y, axis=0)
    y_ecart_type = np.std(y, axis=0)
    return x, y_moyenne, y_ecart_type

def afficher_graphique(x, y_moyenne, y_ecart_type):
    if x is None or y_moyenne is None or y_ecart_type is None:
        return
    
    plt.plot(x, y_moyenne)
    plt.fill_between(x, y_moyenne - y_ecart_type, y_moyenne + y_ecart_type, 
                     color="b", alpha=0.2, label="Écart-type")
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    plt.title('Les reward moyenne sur 10 seed')
    plt.show()

def main():
    chemin_fichier = '../docs/DSACMeanReward.txt'  # remplacez par le chemin de votre fichier
    try:
        donnees = lire_fichier(chemin_fichier)
        x, y_moyenne, y_ecart_type = traiter_donnees(donnees)
        afficher_graphique(x, y_moyenne, y_ecart_type)
    except FileNotFoundError:
        print(f"Le fichier {chemin_fichier} n'existe pas.")

if __name__ == "__main__":
    main()
