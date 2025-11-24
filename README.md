# TP4 — Experiment Tracking avec MLflow (YOLO Tiny)

## 🎯 Objectifs
Ce TP a pour objectif de mettre en place un pipeline complet de suivi d’expériences (experiment tracking) avec MLflow appliqué à un modèle YOLO Tiny entraîné sur un mini-dataset dérivé de COCO128 (1 classe : `person`).

Vous apprendrez à :
- Tracer plusieurs expériences via MLflow (baseline + grille d'hyperparamètres)
- Comparer les résultats (mAP, précision, rappel)
- Inspecter les artefacts (confusion matrix, courbes d’entraînement)
- Documenter une décision de promotion de modèle
- (Optionnel) Publier un modèle dans le Model Registry (Staging / Production)
## 📦 Structure du projet

```bash
mlflow-cv-yolo/
├── data/
│ └── tiny_coco/ # Mini-dataset généré via script
├── reports/
│ └── templates/
│ └── decision_template.md # Gabarit du rapport décisionnel
├── scripts/
│ ├── run_grid.ps1 # Grille de 8 runs (Windows PS)
│ ├── run_grid.cmd # Grille de runs CMD
├── src/
│ └── train_cv.py # Script d'entraînement YOLO + MLflow
├── tools/
│ └── make_tiny_person_from_coco128.py
├── docker-compose.yml # MLflow + MinIO
├── mlflow.env # Variables d'env MLflow/MinIO
├── requirements.txt
└── README.md
```
## 🛠 1. Installation & Préparation

### ✔️ Cloner VOTRE fork
```bash
git clone <URL_DE_VOTRE_FORK>.git
cd mlflow-cv-yolo
```
### ✔️ Créer l’environnement Python (Windows)
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
## 📂 2. Génération du mini-dataset + Tracking DVC
python tools/make_tiny_person_from_coco128.py
### Tracking DVC :
```bash
dvc init
dvc add data/tiny_coco -R
git add data/tiny_coco.dvc .gitignore .dvc/ .gitattributes
git commit -m "Track dataset tiny_coco with DVC"
```
## 🚀 3. Lancer MLflow + MinIO (Docker)
docker compose up -d

# Table of contents  
1. UI MLflow : [http://localhost:5000](#http://localhost:5000)  
2. MinIO Console : [http://localhost:9001](#http://localhost:9001)  
user: minioadmin ****
pass: minioadmin

## 🔗 4. Configuration MLflow (Tracking Server)
set MLFLOW_TRACKING_URI=http://localhost:5000
## 🧪 5. Lancer le baseline
python -m src.train_cv --epochs 3 --imgsz 320 --exp-name cv_yolo_tiny
## 🔁 6. Lancer la grille de runs (8 expériences)
scripts\run_grid.cmd

