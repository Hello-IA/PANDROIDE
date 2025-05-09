import matplotlib.pyplot as plt
import numpy as np

def lire_fichier(chemin):
    with open(chemin, 'r') as fichier:
        lignes = fichier.readlines()
        donnees = []
        for ligne in lignes:
            ligne = ligne.strip().replace('[', '').replace(']', '')
            valeurs = [float(val) for val in ligne.split(',')]
            donnees.append(np.array(valeurs))
        return donnees

def traiter_donnees(donnees):
    
    x = donnees[0]
    nb_step = x.shape[0]
    y = donnees[1:]
    for i in range(len(y)):
        y[i] = y[i][:nb_step]
    y = np.array(y)
    
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
    plt.title('Les reward moyenne sur 10')
    plt.show()

def main():
    chemin_fichier = '../docs/DSACMeanReward1env.txt'  
    try:
        donnees = lire_fichier(chemin_fichier)
        x, y_moyenne, y_ecart_type = traiter_donnees(donnees)
        afficher_graphique(x, y_moyenne, y_ecart_type)
    except FileNotFoundError:
        print(f"Le fichier {chemin_fichier} n'existe pas.")

if __name__ == "__main__":
    main()
