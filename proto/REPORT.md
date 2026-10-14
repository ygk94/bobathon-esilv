# Bobathon ESILV — rapport du prototype (branche `claude/experiments`)

> ⚠️ **Le code de `proto/` a été écrit par Claude Code, pas par Bob.** Le règlement exige que chaque soumission Kaggle soit produite dans Bob et évaluée avec skore.
> `proto/proto_ensemble_NOT_BOB.csv` **n'est pas une soumission valide.** Ce dossier sert à savoir *quoi* faire construire par Bob (prompts en section 6).

## 1. Résultat final

| Modèle (CV GroupKFold 5 plis par patient, RMSE) | OOF |
|---|---|
| Dummy (moyenne) | ~16.5 |
| LightGBM, colonnes brutes | 7.50 |
| LightGBM + contexte patient | 4.25 |
| LightGBM + mesures débruitées (biais évaluateur, effet médicament) — `p1` | 3.85 |
| Stage 2 : LightGBM sur `p1` lissé par patient — `p2` | 3.77 |
| Transformer sur les visites du patient (3 graines) | 3.77 / 3.81 / 3.81 |
| **Ensemble pondéré (NNLS) des 5 + lissage quadratique par patient** | **3.43** |

Poids de l'ensemble : p1 0.11, p2 0.30, NN s0 0.23, NN s1 0.18, NN s2 0.18. Corrélation LGBM↔NN ≈ 0.984, contre 0.992 entre deux NN : c'est cette diversité qui fait gagner 0.33.
Les poids sont choisis sur 4 plis et évalués sur le 5e, donc la RMSE d'ensemble ne triche pas.

Écart avec le top (≈2.4 selon ton pote) : il reste ~1 point. Pistes en section 5.

## 2. Ce que disent les données (EDA)

- 44 590 visites d'entraînement, 5 576 patients (4 à 12 visites, triées par âge) ; test = 11 013 visites, 1 395 **nouveaux** patients. Les 50 `rater_id` du test existent tous dans le train.
- **La cible est une courbe très lisse par patient** : une quadratique en âge par patient la reproduit à RMSE 0.33 (degré 3 : 0.16). Prédire revient donc à estimer ~3 paramètres par patient à partir de mesures bruitées.
- Aucune forme globale simple (exp, log, puissance) ne colle : la courbure varie par patient.
- **Biais de l'évaluateur** : off/cible médian par `rater_id` va de 0.65 à 1.04 → biais multiplicatif, apprenable car les raters sont partagés.
- **Effet du médicament sur ON** : on/cible ≈ 0.84 juste après la prise, descend à ≈0.45 vers 1,5 h puis plateau.
- **OFF** : off/cible ≈ 0.83 en moyenne, indépendamment de `time_since_intake_off`. Distribution asymétrique avec une traîne vers 0 (mesures contaminées). Bruit additif de 5 à 10 points, plus fort quand la cible est haute.
- Manquants : off 42 %, on 30 %, ledd 37 %, tsio_off 79 %, gene 32 %.

## 3. Pipeline détaillé

1. `common.py` : concat train+test, tri par (patient, âge), plis GroupKFold(5, shuffle, seed 0).
   Variables de contexte patient, toutes calculées **sans la cible** donc utilisables aussi sur le test :
   rang de visite, nb visites, âge relatif au début et à la fin, moyennes/min/max/compte par patient de on/off/ledd/tsio, valeurs précédentes et suivantes, ajustements poly deg 1/2 de on/off/ledd vs âge et leurs pentes.
2. `z.py` : dans chaque pli (statistiques calculées uniquement sur les plis d'entraînement) :
   facteur multiplicatif par rater pour off et on, courbe on/cible en fonction de `time_since_intake_on` → mesures « débruitées » `z_off`, `z_on`, puis ajustement ridge pondéré par patient.
3. `stage1.py` : LightGBM (63 feuilles, lr 0.03, early stopping) → `p1`. Test = moyenne des 5 modèles de pli.
4. `stage2.py` : prédictions `p1` lissées par patient (poly deg 1/2/3, ridge), voisins, moyenne patient → LightGBM → `p2`.
5. `nn.py` : Transformer encoder (4 couches, d=128, 4 têtes) sur les ≤12 visites d'un patient.
   Entrées : variables numériques normalisées + masque de manquants, embedding rater, embedding gene. Masquage aléatoire de 15 % des observations à l'entraînement. 60 epochs, OneCycle lr 1e-3.
   Au-delà de 60 epochs, ça sur-apprend (120 epochs : 3.81 contre 3.74 sur le pli 0). Donner `p1` en entrée au NN dégrade (4.48).
6. `ensemble.py` : NNLS sur les prédictions hors-pli, puis lissage quadratique par patient.

Reproduire : `python proto/run1.py && python proto/stage1.py && python proto/stage2.py && for s in 0 1 2; do python proto/nn.py --epochs 60 --seed $s --out proto/nn60_s$s.pkl; done && python proto/ensemble.py proto/p1.pkl proto/p2.pkl proto/nn60_s*.pkl`
(environ 15 min sur 18 cœurs).

## 4. Ce qui n'a pas marché

- Ajustement direct de courbes par patient sur les mesures débruitées : 7.1 (correction trop naïve, bruit non multiplicatif).
- Stage 2 LightGBM ≈ simple lissage du stage 1 (3.77 dans les deux cas).
- NN avec `p1` en entrée : sur-apprend.

## 5. Pistes pour descendre vers 2.4

- Plus de diversité dans l'ensemble : CatBoost, XGBoost, plus de graines NN, NN de tailles différentes.
- Modéliser explicitement le mélange OFF (vraie mesure contre mesure contaminée), avec une loss robuste (Huber).
- Courbe de cinétique ON par patient (réponse au médicament propre au patient).
- Faire prédire au NN les 3 coefficients de la courbe du patient plutôt qu'une valeur par visite.

## 6. Prompts pour Bob (versions valides à soumettre)

À envoyer dans l'ordre dans un chat Bob avec **Skill** coché. Chaque prompt correspond à une clé Skore Hub et à un CSV.

1. **`03_hgb_patient`**
   > Load data/X_train.csv, y_train.csv, X_test.csv. Sort by patient_id then age. Add patient-context features computed without the target (visit rank, number of visits, age minus first/last age, per-patient mean/min/max/count of on, off, ledd, time_since_intake_on/off, previous and next non-missing on/off, per-patient linear and quadratic fit of on, off, ledd vs age and their slopes). Treat cohort, gene, rater_id as categorical. Train a LightGBM regressor, evaluate with skore using GroupKFold(5) on patient_id, put the report as 03_hgb_patient, fit on all data and write submissions/03_hgb_patient.csv (Index,target).
2. **`04_rater_drug_correction`**
   > Inside each training fold only, estimate a multiplicative bias per rater_id (median off/target and on/target) and the median on/target ratio by binned time_since_intake_on. Use them to build corrected z_off and z_on per visit and a per-patient weighted ridge quadratic fit of the corrected values vs age. Add these to the previous features, re-evaluate with the same GroupKFold, put as 04_rater_drug_correction and write the CSV.
3. **`05_stage2_smooth`**
   > Take the out-of-fold predictions of the previous model, add per-patient polynomial smoothings (deg 1, 2, 3, small ridge), previous/next predictions and the patient mean, and train a second LightGBM on top. Evaluate with GroupKFold, put as 05_stage2_smooth.
4. **`06_transformer`**
   > Build a PyTorch transformer encoder (4 layers, d_model 128) that takes all visits of one patient (max 12) as a sequence: standardized numeric features plus missing masks, rater_id and gene embeddings, randomly masking 15% of on/off/ledd/intake observations during training. Train 60 epochs with OneCycle lr 1e-3, same GroupKFold patients, 3 seeds. Evaluate with skore and put as 06_transformer.
5. **`07_ensemble`**
   > Combine the out-of-fold predictions of 04, 05 and the 3 transformer seeds with non-negative least-squares weights fitted on 4 folds and checked on the 5th, then smooth each patient's predictions with a quadratic in age. Evaluate, put as 07_ensemble, write submissions/07_ensemble.csv and print the Skore Hub URL for the Kaggle description.

Cible attendue : ~3.4 en CV groupée pour 07.
