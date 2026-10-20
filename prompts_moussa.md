# Prompts utilisés — partie modèle (Moussa)

À fusionner dans `prompts.md` (Loan). Pour chaque prompt : ce que l'IA a proposé, ce qu'on a vérifié nous-mêmes, ce qu'on a corrigé ou refusé.

---

## Prompt 1 — Le diagnostic avant de coder (diapo 24 du cours)

> « Tu es mon reviewer ML. Voici : question, grain, cible, colonnes, EDA, date de prédiction. 1. La cible est-elle définie ? 2. Quelles features sont disponibles à temps ? 3. Quelles fuites ou biais possibles ? 4. Quel split et quelle baseline ? 5. Quelle métrique et pourquoi ? Pose 3 questions bloquantes avant de coder. »

**Ce que l'IA a relevé :**
- la cible (les buts) mélange 90 et 120 minutes à cause des prolongations ;
- une petite fuite possible dans l'Elo « maison recalé » (droite ajustée sur toute la saison) ;
- le risque le plus sérieux : le test 2023-2025 avait déjà servi à décider d'ajouter la variable « nouveau format ».

**Ses 3 questions bloquantes, et nos réponses :**
1. Forme récente : attendre Loan ? → Non : une section qui se branche quand son fichier arrive, et un pré-test rapide en attendant.
2. Le test déjà « vu » ? → On le garde pour le modèle choisi avant (Elo + terrain), on justifie le format uniquement par le test vers l'avant (2025-26), et on l'écrit comme limite.
3. Forcer des nuls dans la matrice de confusion ? → Non : règle « issue la plus probable », car la simulation n'utilise que les probabilités.

**Ce qu'on a vérifié nous-mêmes :**
- les prolongations : **21 matchs sur 1 871 (1,1 %)**, 22 buts sur 5 735 → effet négligeable, écrit comme limite ;
- l'Elo recalé : 81 lignes club-match sur 3 742 (2,2 %). Loan l'a confirmé de son côté.

---

## Prompt 2 — Un pipeline vérifiable (diapo 25)

> « Génère un notebook scikit-learn minimal : split adapté, pas de fuite, preprocessing en Pipeline ; baseline Dummy, modèle simple, random_state fixé ; métriques adaptées et tableau comparatif ; cellules Markdown qui expliquent chaque étape ; aucun téléchargement externe ni donnée inventée. »

**Ce que l'IA a produit :** `MODEL_POISSON.ipynb` v2, avec un `Pipeline` (`SimpleImputer` + `PoissonRegressor`), un split chronologique, `DummyClassifier` et `DummyRegressor`, et les métriques log loss, RPS, accuracy, balanced accuracy, matrice de confusion et MAE des buts.

**Ce qu'on a vérifié :**
- **Le Pipeline donne le même modèle que statsmodels.** Un `assert` compare les coefficients. Il a d'abord **échoué** : avec ses réglages par défaut, `PoissonRegressor` s'arrête trop tôt (écart de 0,0002). Corrigé avec `solver='newton-cholesky'` et `tol=1e-10` : écart de 10⁻¹¹.
- **Les fonctions exportées** (`probas_match`, `tirer_score`) donnent les mêmes probabilités que le Pipeline (à 10⁻⁹ près), et `tirer_score` retrouve les bons buts moyens sur 100 000 tirages.
- **Tous les chiffres de la v1 sont retrouvés à l'identique** avec le Pipeline.

**Ce qu'on a corrigé :** des phrases causales (« le nouveau format marque plus de buts ») remplacées par des observations (« depuis le nouveau format, on observe plus de buts »).

---

## Prompt 3 — L'IA critique son propre résultat (diapo 26)

> « Agis comme un auditeur hostile. Cherche : data leakage et cible mal définie, train/test contaminés, métrique trompeuse, classes déséquilibrées, faible taille d'échantillon, features non disponibles en production. Propose 3 tests concrets pour réfuter le modèle. »

**Les tests, et ce qu'ils ont donné (test 2023-2025, modèle Elo + terrain) :**

| Test | Résultat | Conclusion |
|---|---|---|
| **Cible mélangée** : on mélange les scores de l'apprentissage | log loss 1,026 ± 0,011, contre 1,020 pour le Dummy (0,902 pour le vrai modèle) | ✅ sans le vrai lien entre Elo et scores, le modèle ne vaut plus rien : pas de fuite par la structure des données |
| **Retirer l'Elo « maison recalé »** (la petite fuite) | log loss 0,9050 dans les deux cas, avec ou sans ces matchs à l'apprentissage | ✅ la fuite n'a aucun effet mesurable |
| **Sous-groupes** | gain sur le Dummy : +0,14 en phase de groupes, seulement +0,03 en élimination directe | ⚠️ voir ci-dessous |
| **Calibration des favoris** | annoncé à 65 % → gagne 75 % ; annoncé à 74 % → 77 % ; annoncé à 87 % → 91 % | ⚠️ sur 2023-2025, le modèle est un peu **trop prudent** avec les favoris |

**Ce que les tests ont révélé :**
1. **Le modèle paraît moins bon en élimination directe.** On a vérifié pourquoi : à écart d'Elo égal (moins de 150 points), son gain est le même en phase de groupes (+0,029) qu'en élimination directe (+0,027). Les matchs à élimination directe sont simplement plus **serrés** (113 points d'écart moyen, contre 149) : ils sont plus durs à prévoir pour n'importe quel modèle. Ce n'est pas un défaut propre au modèle, mais c'est important, car ce sont ces matchs qui décident du titre.
2. **Les favoris gagnent plus que prévu sur 2023-2025.** C'est cohérent avec le nouveau format (plus de buts, moins de nuls, donc des résultats plus tranchés), que la variable « format » corrige en partie. Attention : à l'échelle du tournoi, le backtest montrait l'inverse (un peu trop confiant sur les grands favoris). Les deux ne se contredisent pas : un match isolé et un parcours de 4 tours ne posent pas la même question.
3. **Les matchs avec un Elo estimé sont les mieux prédits** (gain de +0,34), parce que ce sont souvent des matchs très déséquilibrés contre de petits clubs. L'estimation ne dégrade donc pas les prédictions.
