# 🎫 Ticket de sortie

### 1. Question et cible
Prédire les buts de chaque équipe dans chaque match de C1, pour simuler la saison 2026-27 et donner une probabilité de titre à chaque club.

### 2. Baseline et métrique
Un Dummy (classe majoritaire, ou moyenne des buts). Le modèle est choisi sur la log loss ; on rapporte aussi l’accuracy et la MAE des buts.

### 3. Scores apprentissage, validation et test
58,8 %, 59,7 % et 60,4 %, au niveau de Pinnacle (60,2 %).

### 4. Un risque identifié à l’audit
En élimination directe, le gain n’est pas démontré sur le test (118 matchs), alors que c’est là que se joue le titre.

### 5. La prochaine expérience, sans toucher au test
Corriger la hausse des buts et la surestimation des nuls, et la valider en validation glissante seulement.
