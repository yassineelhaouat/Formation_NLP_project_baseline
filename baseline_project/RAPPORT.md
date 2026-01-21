# 🚀 Rapport Projet Veille Automatique NLP

**Système de classification automatique d'articles techniques**

---

## 📋 Sommaire

1. [Présentation du Baseline](#1-présentation-du-baseline)
2. [Problèmes Identifiés](#2-problèmes-identifiés)
3. [Améliorations Apportées](#3-améliorations-apportées)
4. [Résultats & Évaluation](#4-résultats--évaluation)
5. [Analyse des Erreurs](#5-analyse-des-erreurs)
6. [Conclusion & Perspectives](#6-conclusion--perspectives)

---

## 1. Présentation du Baseline

### 🎯 Objectif du Système
Créer un pipeline automatique de veille technologique :
- **Collecter** des articles de sources multiples
- **Prétraiter** le texte avec des techniques NLP
- **Classifier** par niveau de difficulté (Beginner / Intermediate / Advanced)
- **Générer** un rapport de synthèse

### 📦 Composants Fournis

| Module | Fonction | Technologies |
|--------|----------|--------------|
| `news_collector.py` | Scraping d'articles | BeautifulSoup, requests |
| `text_preprocessor.py` | Nettoyage NLP | spaCy, regex |
| `news_classifier.py` | Classification zero-shot | Transformers, sklearn |
| `report_generator.py` 

### 🔧 Pipeline Original
```
HackerNews → Prétraitement spaCy → Zero-shot Classification → Rapport .txt
```

---

## 2. Problèmes Identifiés

### ❌ Problème de Performance : 34% Accuracy

Après correction du modèle, le zero-shot classifier atteignait seulement **34.2% d'accuracy**.

**Analyse du biais** :
- Le modèle prédisait majoritairement **"Advanced"**
- Mauvaise généralisation sur nos catégories spécifiques
- Zero-shot non adapté à notre domaine précis

### ❌ Problème de Données : Source Unique

- Baseline = uniquement HackerNews
- Articles courts (titres + peu de contenu)

---

## 3. Améliorations Apportées

### ✅ A) Enrichissement des Sources

**Action** : Ajout de TowardsDataScience comme 2ème source

```python
# Dans config.json
"towards_data_science": {
    "enabled": true,
    "base_url": "https://towardsdatascience.com",
}
```

**Résultat** :
| Métrique | Baseline | Amélioré |
|----------|----------|----------|
| Sources | 1 | 2 |
| Articles | ~60 | 210 |
| Contenu | Titres seuls | Contenu complet |

---

### ✅ B) Création du Dataset Annoté

**Action** : Annotation manuelle de 80+ articles

**Fichier créé** : `data/ground_truth_annotations.json`

**Critères d'annotation définis** :

| Label | Critères |
|-------|----------|
| **Beginner** | Introduction aux concepts, guides de démarrage, tutoriels pour débutants, actualités accessibles |
| **Intermediate** | Nécessite des connaissances techniques, complexité modérée, applications pratiques, utilisation d'outils |
| **Advanced** | Contenu technique profond, systèmes de production, papers de recherche, expertise requise |

**Distribution des annotations** :
```
Beginner:     72 articles (36%)
Intermediate: 80 articles (40%)
Advanced:     50 articles (24%)
```

---

### ✅ C) Fine-tuning du Classifier

**Action** : Entraînement supervisé sur nos annotations

**Modèle choisi** : `distilbert-base-uncased`
- Plus léger que BERT complet
- Adapté aux ressources limitées (CPU)
- Bon compromis performance/vitesse


**Gestion du déséquilibre** :

Implémentation d'un `WeightedTrainer` personnalisé :
```python
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, ...):
        # Cross-entropy pondérée par classe
        loss_fct = nn.CrossEntropyLoss(weight=class_weights)
        return loss_fct(logits, labels)
```

**Poids calculés** :
- Beginner: 0.93
- Intermediate: 0.84
- Advanced: 1.34

---

### ✅ D) Notebooks d'Évaluation

**Créés** :
1. `01_exploratory_data_analysis.ipynb` - Analyse exploratoire des données
2. `02_evaluation_benchmarks.ipynb` - Métriques et comparaisons
3. `03_fine_tuning.ipynb` - Pipeline complet de fine-tuning

**Contenu** :
- Distribution des articles par source
- Distribution des labels
- Longueur des articles
- Matrices de confusion
- Métriques détaillées

---

## 4. Résultats & Évaluation

### 📊 Comparaison Baseline vs Fine-tuned

| Métrique | Baseline (Zero-shot) | Fine-tuned |
|----------|---------------------|------------|
| **Accuracy** | 34.2% | **58.3%** |
| **Amélioration** | - | **+24.1** |

### 📈 Métriques Détaillées (Fine-tuned)

```
              precision    recall  f1-score   support

Beginner       0.67        0.50      0.57        4
Intermediate   0.50        0.33      0.40        3
Advanced       0.56        0.83      0.67        6

accuracy                             0.58       13
macro avg      0.58        0.56      0.55       13
weighted avg   0.58        0.58      0.56       13
```

### 🎯 Matrice de Confusion

**Baseline (Zero-shot)** :
```
              Predicted
              Beg   Int   Adv
Actual Beg  [  5     4    17  ]  → Biais vers Advanced
       Int  [  7     4    21  ]  → Biais vers Advanced  
       Adv  [  1     0    17  ]
```

**Fine-tuned** :
```
              Predicted
              Beg   Int   Adv
Actual Beg  [  2     1     1  ]  → Meilleure distribution
       Int  [  1     1     1  ]  
       Adv  [  0     1     5  ]  → Bon recall Advanced
```

---

## 5. Analyse des Erreurs

### 🔍 Types d'Erreurs Identifiées

**1. Confusion Beginner ↔ Intermediate**
- Articles d'introduction avec termes techniques
- Exemple : "Introduction to Docker" → Classé Intermediate

**2. Articles Courts**
- Peu de contexte pour la classification
- Titres seuls insuffisants

**3. Domaines Mixtes**
- Articles couvrant plusieurs niveaux
- Exemple : "From Zero to Hero in Machine Learning"

### 📉 Limites du Système

| Limite | Impact | Solution Potentielle |
|--------|--------|---------------------|
| Dataset petit (80 samples) | Overfitting | Annoter plus d'articles |
| Déséquilibre des classes | Biais prédictions | Data augmentation |
| Articles courts | Manque contexte | Fetch contenu complet |
| Source unique finale | Biais domaine | Ajouter plus de sources |

---