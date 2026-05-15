# RAG-time / LogiStore

> Moteur de recherche d'entreprise basé sur un RAG hybride (BM25 + sémantique), centré sur les tickets de support de l'entreprise fictive LogiStore.

**Projet académique** — M2 IA & Data Science, La Plateforme Marseille — Mai 2026
**Auteur** — Mohamed AHMEDVALL

---

## 🎯 Objectif

Construire un moteur de recherche qui démontre la valeur d'un système RAG appliqué à un contexte d'entreprise, en commençant par les tickets de support et en posant les fondations pour une extension future (ERP, CRM, PIM, GED).

## 🏗️ Stack

- **Qdrant** — vector store self-hosted (hybride dense + sparse natif)
- **BGE-M3** — embeddings multilingues (dense + sparse en 1 forward)
- **BGE-reranker-v2-m3** — cross-encoder optionnel pour le post-rerank
- **FastAPI** — backend REST
- **Streamlit** — frontend MVP
- **OpenRouter** — LLM pour la synthèse optionnelle (Claude / Mistral)
- **RAGAS + LLM-as-judge** — évaluation

Voir `docs/02_architecture.md` pour le détail et `docs/architecture.png` pour le schéma.

## 🚀 Démarrage rapide

```bash
# Prérequis : Docker, Python 3.11+, make

# 1. Installation
make install

# 2. Configuration
cp .env.example .env
# éditer .env (clé OpenRouter notamment)

# 3. Démarrage de Qdrant
make qdrant-up

# 4. Téléchargement du dataset
make download-data

# 5. Ingestion (~10-20 min selon volume)
make ingest

# 6. Lancement de l'API et du frontend (deux terminaux)
make api    # http://localhost:8000
make ui     # http://localhost:8501
```

## 📊 Évaluation

```bash
make eval
# Produit un rapport dans data/evaluation/report.html
```

Métriques calculées : Precision@5, Recall@10, NDCG@10, MRR + RAG Triad via LLM-as-judge.

## 📁 Documentation

- [`docs/01_veille_technologique.md`](docs/01_veille_technologique.md) — état de l'art RAG
- [`docs/02_architecture.md`](docs/02_architecture.md) — architecture cible
- [`docs/03_etude_risques.md`](docs/03_etude_risques.md) — analyse de risques
- [`docs/04_evaluation_protocole.md`](docs/04_evaluation_protocole.md) — protocole d'évaluation

## 🧪 Tests

```bash
make test         # tests unitaires
make test-int     # tests d'intégration (Qdrant requis)
make lint         # ruff + black --check
```

## 📜 Licence

MIT — projet académique
