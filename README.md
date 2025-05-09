# Projet M1 - Implémentation et Étude de Discrete Soft Actor-Critic avec BBRL

Ce projet a été réalisé dans le cadre de l'UE Projet ANDROIDE du Master 1 Informatique (parcours AI2D) à Sorbonne Université.

## Objectif

L’objectif du projet est de :
- Prendre en main la bibliothèque `BBRL` (Black-Box Reinforcement Learning),
- Implémenter plusieurs algorithmes de Deep Reinforcement Learning classiques (DQN, DDQN, DDPG, TD3, SAC),
- Réaliser une version **discrète** de l'algorithme **Soft Actor-Critic (DSAC)**,
- Étudier expérimentalement le comportement de l'actor et du critic.

## Structure du projet

- DDPG/, DQN/, SAC/, TD3Discret/
    -Contiennent des notebooks explicatifs pour chaque algorithme implémenté, permettant de mieux comprendre leur fonctionnement et leur entraînement avec BBRL.
- DSAC/
Contient l’implémentation complète et les expériences menées sur Discrete Soft Actor-Critic (DSAC) :
    - src/ : Fichiers source Python de l’implémentation.
    - docs/ : Résultats numériques (logs, récompenses, meilleurs hyperparamètres, etc.).
    - outputs/ : Répertoires générés par BBRL (logs, modèles, etc.).
    - plot/ : Graphiques et figures issues des études expérimentales.
  
