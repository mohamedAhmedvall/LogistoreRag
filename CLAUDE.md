# CLAUDE.md — RAG-time / LogiStore

**Tu (Claude Code) travailles sur le projet RAG-time / LogiStore.** Ce fichier définit le contexte permanent et les conventions à respecter sur l'ensemble du projet. Lis-le intégralement avant la première action.

---

## 1. Mission du projet

Construire un MVP de moteur de recherche d'entreprise basé sur un système RAG hybride, centré sur les tickets de support de l'entreprise fictive LogiStore. Le système doit :

- Indexer un dataset de tickets de support en combinant recherche lexicale et sémantique
- Permettre une recherche hybride avec filtres metadata et ranking explicable
- Offrir optionnellement une synthèse LLM des résultats top-K
- Exposer une API REST claire et un frontend Streamlit de consultation
- Inclure un protocole d'évaluation complet (métriques IR + LLM-as-judge)

Le projet est livré dans un cadre académique (M2 IA & Data Science, La Plateforme Marseille) et doit démontrer des compétences techniques substantielles.

---

## 2. Stack technique imposée

**N'introduis aucune dépendance hors de ce stack sans question préalable.**

| Couche | Technologie | Justification |
|---|---|---|
| Vector store | **Qdrant** (Docker self-hosted) | Hybride natif dense+sparse, filtres payload-aware, RGPD-friendly |
| Embeddings | **BGE-M3** (local via `FlagEmbedding` ou `fastembed`) | Multilingue FR/EN, dense + sparse en 1 forward pass |
| Reranker (optionnel) | **BGE-reranker-v2-m3** | Cross-encoder léger multilingue |
| Backend | **FastAPI** | Type-safety Pydantic, performance, doc OpenAPI automatique |
| Frontend | **Streamlit** | Rapidité de mise en place pour le MVP |
| LLM (synthèse optionnelle) | **OpenRouter** (Claude Sonnet ou Mistral) | Flexibilité de modèles, abstraction possible vers Ollama |
| Évaluation | **RAGAS** + LLM-as-judge custom | Standard industrie |
| Données | **Kaggle Customer Support Tickets** (200K+) | Réaliste, multi-catégories |
| Conteneurisation | **Docker Compose** | Reproductibilité |
| Tests | **pytest** | Standard |
| Logging | **structlog** ou **logging** stdlib | Logs structurés JSON |
| Config | **pydantic-settings** + `.env` | Type-safe configuration |

**Python 3.11+** uniquement. Pas de Python 3.8/3.9/3.10.

---

## 3. Arborescence du repo

L'arborescence est imposée et doit être respectée scrupuleusement :

```
ragtime-logistore/
├── CLAUDE.md                    # ce fichier (ne pas modifier)
├── README.md                    # documentation utilisateur
├── pyproject.toml               # config projet + deps
├── requirements.txt             # deps gelées
├── docker-compose.yml           # Qdrant + services
├── Makefile                     # commandes courantes
├── .env.example                 # template de config
├── .gitignore
│
├── docs/                        # livrables documentaires
│   ├── 01_veille_technologique.md
│   ├── 02_architecture.md
│   ├── architecture.png
│   ├── 03_etude_risques.md
│   └── 04_evaluation_protocole.md
│
├── src/
│   └── ragtime/
│       ├── __init__.py
│       ├── config.py            # settings Pydantic
│       ├── logging_setup.py     # config logging structuré
│       ├── models.py            # schémas Pydantic (Ticket, SearchQuery, SearchResult, ...)
│       │
│       ├── ingest/              # pipeline d'ingestion
│       │   ├── __init__.py
│       │   ├── loader.py        # chargement dataset Kaggle
│       │   ├── normalizer.py    # nettoyage / normalisation (bronze → silver)
│       │   ├── chunker.py       # stratégie de chunking pour les tickets
│       │   └── pipeline.py      # orchestration bronze → silver → gold
│       │
│       ├── embeddings/          # gestion des embeddings
│       │   ├── __init__.py
│       │   ├── bge_m3.py        # wrapper BGE-M3 (dense + sparse)
│       │   └── reranker.py      # wrapper BGE-reranker
│       │
│       ├── index/               # interactions Qdrant
│       │   ├── __init__.py
│       │   ├── client.py        # client Qdrant + gestion collections
│       │   ├── schema.py        # définition des collections (vectors config, payload schema)
│       │   └── indexer.py       # upsert batch avec idempotence
│       │
│       ├── search/              # logique de recherche
│       │   ├── __init__.py
│       │   ├── hybrid.py        # recherche hybride RRF
│       │   ├── filters.py       # construction des filtres Qdrant
│       │   └── reranking.py     # post-rerank optionnel
│       │
│       ├── llm/                 # accès LLM
│       │   ├── __init__.py
│       │   ├── provider.py      # interface abstraite LLMProvider
│       │   ├── openrouter.py    # implémentation OpenRouter
│       │   └── synthesis.py     # prompt + génération de synthèse avec citations
│       │
│       ├── api/                 # FastAPI app
│       │   ├── __init__.py
│       │   ├── main.py          # app FastAPI + middlewares
│       │   ├── routes/
│       │   │   ├── __init__.py
│       │   │   ├── search.py
│       │   │   ├── ingest.py
│       │   │   ├── synthesize.py
│       │   │   ├── eval.py
│       │   │   └── health.py
│       │   └── dependencies.py  # DI FastAPI
│       │
│       └── evaluation/          # évaluation
│           ├── __init__.py
│           ├── golden_dataset.py # 30 requêtes de référence
│           ├── metrics.py       # Precision@K, Recall@K, NDCG, MRR
│           ├── llm_judge.py     # LLM-as-judge avec position swapping
│           └── runner.py        # exécution d'une campagne d'éval
│
├── app/                         # frontend Streamlit
│   ├── streamlit_app.py         # point d'entrée
│   ├── pages/
│   │   ├── 1_🔍_Recherche.py
│   │   ├── 2_📊_Évaluation.py
│   │   └── 3_⚙️_Admin.py
│   └── components/
│       ├── search_bar.py
│       ├── result_card.py
│       └── filters_sidebar.py
│
├── scripts/                     # scripts utilitaires
│   ├── download_dataset.py      # téléchargement Kaggle
│   ├── run_ingestion.py         # lancement ingestion complète
│   ├── run_evaluation.py        # lancement évaluation
│   └── reset_qdrant.py          # purge collection (dev only)
│
├── tests/                       # tests pytest
│   ├── __init__.py
│   ├── conftest.py              # fixtures
│   ├── unit/
│   │   ├── test_chunker.py
│   │   ├── test_embeddings.py
│   │   ├── test_hybrid_search.py
│   │   └── test_metrics.py
│   └── integration/
│       ├── test_api_search.py
│       └── test_ingestion_pipeline.py
│
├── data/                        # données (gitignore)
│   ├── raw/                     # bronze
│   ├── processed/               # silver / gold
│   └── golden/
│       └── queries.json         # golden dataset
│
└── notebooks/                   # exploration (gitignore le data)
    └── 01_data_exploration.ipynb
```

---

## 4. Conventions de code

**Style.** Black + Ruff. Configuration dans `pyproject.toml`. Ligne max 100 caractères.

**Typage.** Type hints **partout**. Pydantic pour tous les modèles de données et settings. Pas de `dict[str, Any]` sauf à l'interface I/O brute.

**Async.** FastAPI en async natif. Embeddings et appels LLM en async pour parallélisation.

**Logs.** `structlog` avec contexte (request_id, user, latencies). Pas de `print()` en code applicatif.

**Erreurs.** Exceptions custom héritant de `RAGtimeError` (`IngestionError`, `IndexError`, `SearchError`, `LLMError`). Pas de `except Exception: pass`.

**Tests.** Couverture > 70% sur `src/ragtime/`. Tests unitaires pour la logique pure (chunker, métriques, filtres), tests d'intégration pour les pipelines complets avec Qdrant in-memory ou test container.

**Docstrings.** Style Google. Toutes les fonctions publiques documentées.

**Imports.** Absolus depuis `ragtime`. Pas d'imports relatifs `from ..foo`.

**Configuration.** Tout passe par `config.py` (Pydantic Settings). Aucune valeur hardcodée pour : URLs, chemins, noms de collection, modèles, paramètres HNSW, seuils de pertinence.

---

## 5. Contraintes critiques

**Souveraineté des données.** Les embeddings sont calculés **localement** avec BGE-M3. Aucune donnée de ticket n'est envoyée à un embedding model externe. Le LLM externe (OpenRouter) n'est appelé que pour la synthèse explicite, sur des passages déjà retrouvés.

**Idempotence.** L'ingestion doit être idempotente : relancer l'ingestion sur les mêmes données ne crée pas de doublons. Utiliser des IDs déterministes (hash du contenu).

**Pas de mock en production.** Le code de production ne contient aucun mock ni stub. Les mocks vivent dans `tests/`.

**Pas de notebook en chemin critique.** Les notebooks sont pour l'exploration, pas pour l'exécution du système. Toute logique réutilisable doit migrer dans `src/ragtime/`.

**Secrets.** Aucun secret en dur. Tout passe par variables d'environnement chargées via `pydantic-settings`. `.env` est `.gitignore`. `.env.example` documente les variables sans valeurs.

**Pas de réseau pendant les tests.** Les tests unitaires ne doivent dépendre d'aucun service externe. Les tests d'intégration utilisent Qdrant in-memory ou un test container marqué `@pytest.mark.integration`.

---

## 6. Commandes Make standardisées

Le `Makefile` expose les commandes courantes :

```
make install        # installe les deps
make qdrant-up      # démarre Qdrant via docker-compose
make qdrant-down    # arrête Qdrant
make download-data  # télécharge le dataset Kaggle
make ingest         # exécute l'ingestion complète
make api            # lance l'API FastAPI en dev (uvicorn --reload)
make ui             # lance le frontend Streamlit
make test           # tests unitaires
make test-int       # tests d'intégration (nécessite Qdrant up)
make eval           # lance la campagne d'évaluation
make lint           # ruff + black --check
make format         # black + ruff --fix
make clean          # nettoie les artefacts
```

Toute nouvelle commande utile doit être ajoutée au Makefile, jamais documentée uniquement dans le README.

---

## 7. Ordre de développement recommandé

Quand tu démarres un nouveau bout de fonctionnalité, suis cet ordre :

1. **Schémas Pydantic** dans `models.py` (contrat avant implémentation)
2. **Tests unitaires** de la logique pure (TDD léger)
3. **Implémentation** de la logique
4. **Câblage API/UI** par-dessus
5. **Tests d'intégration**
6. **Documentation** (docstrings + README si user-facing)

Le pipeline global recommandé est :

1. `config.py` + `logging_setup.py` + `models.py`
2. `ingest/loader.py` (charger le dataset)
3. `ingest/normalizer.py` + `ingest/chunker.py`
4. `embeddings/bge_m3.py` (test sur petit échantillon avant batch complet)
5. `index/client.py` + `index/schema.py` + `index/indexer.py`
6. Script `run_ingestion.py` end-to-end
7. `search/hybrid.py` + `search/filters.py`
8. `api/routes/search.py` + `api/routes/health.py`
9. Frontend Streamlit basique (recherche + résultats)
10. `search/reranking.py` (optionnel)
11. `llm/provider.py` + `llm/openrouter.py` + `llm/synthesis.py`
12. `api/routes/synthesize.py` + bouton "Synthétiser" dans le frontend
13. `evaluation/golden_dataset.py` + `evaluation/metrics.py`
14. `evaluation/llm_judge.py` + `evaluation/runner.py`
15. `api/routes/eval.py` + page Évaluation Streamlit
16. Polish : observabilité, README, exemples

---

## 8. Pièges à éviter (retour d'expérience)

- **Ne pas lancer l'embedding BGE-M3 sur 200K tickets sans batching.** Le batch size doit être configurable et limiter l'usage mémoire. Pour le MVP, échantillonner 10K-20K tickets suffit largement.
- **Ne pas réembedder à chaque ingestion.** Vérifier l'existence du point (par ID déterministe) avant de recalculer.
- **Ne pas concaténer brutalement tous les champs d'un ticket** pour l'embedding. Distinguer subject / description / resolution pour pouvoir surligner les zones de match.
- **Ne pas oublier les filtres de RBAC** même si non implémentés en MVP : prévoir la mécanique dans `filters.py` pour absorber facilement plus tard.
- **Ne pas négliger les cas vides** : query sans résultats, collection vide, LLM indisponible. Chaque chemin doit retourner une réponse propre.
- **Ne pas faire de synthèse LLM sans contexte** : si aucun résultat n'est trouvé, ne pas appeler le LLM (anti-hallucination structurel).

---

## 9. Comment me poser des questions

Si tu rencontres une ambiguïté ou un choix structurant non couvert ici :

1. Vérifie d'abord dans ce CLAUDE.md, le README et `docs/02_architecture.md`
2. Si la question reste, **demande-moi avant d'implémenter** plutôt que d'inventer
3. Documente ta question et la décision prise dans un commit message ou un fichier `docs/ADR/` (Architecture Decision Records)

---

## 10. Définition du "done" pour le MVP

Le MVP est considéré terminé quand :

- [ ] `make qdrant-up && make download-data && make ingest` exécute sans erreur
- [ ] `make api` lance l'API et `/health` répond 200
- [ ] `make ui` ouvre le frontend, une recherche retourne des résultats classés
- [ ] La synthèse LLM fonctionne sur le top-3
- [ ] `make eval` exécute la campagne et produit un rapport
- [ ] `make test` passe (couverture > 70%)
- [ ] README à jour, captures d'écran, instructions complètes
- [ ] Aucun secret en dur, aucun `print()`, aucun `TODO` critique non documenté
