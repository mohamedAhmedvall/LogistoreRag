# ADR-001 — Substitution du modèle d'embedding BGE-M3

**Date :** 2026-05-15
**Statut :** Accepté
**Auteur :** Mohamed AHMEDVALL (avec Claude Code)

## Contexte

`CLAUDE.md §2` impose **BGE-M3** comme modèle d'embedding, avec deux wrappers possibles : `FlagEmbedding` ou `fastembed`. Le projet est livré dans un cadre académique (M2 IA & Data Science) et la fidélité à l'énoncé compte.

Lors de l'installation de l'environnement (Incrément 3) :

- **fastembed 0.8.0** (dernière version stable en mai 2026) **n'expose plus BGE-M3** dans sa liste de modèles supportés (`TextEmbedding.list_supported_models()`). Aucune référence à `bge-m3` dans le code source du package. Le chemin "BGE-M3 via fastembed" mentionné dans CLAUDE.md n'est donc plus praticable.
- **FlagEmbedding** supporte officiellement BGE-M3, mais nécessite l'écosystème PyTorch complet (transformers, sentencepiece, accelerate). Empreinte d'installation : ~3 Go de dépendances + 2.3 Go pour le modèle. Premier chargement ~10 s. Inférence CPU plus lente que l'équivalent ONNX (50-100 docs/s vs 150-300 docs/s).

Un MVP académique doit être facilement reproductible : installation simple, démarrage rapide. La taille du dataset (10 000 tickets après échantillonnage, ~9 200 docs uniques) ne justifie pas le surcoût opérationnel de PyTorch pour gagner ~1 point MTEB sur un benchmark.

## Décision

Remplacer **BGE-M3** par **`intfloat/multilingual-e5-large`** comme modèle d'embedding dense, et utiliser **BM25** (`Qdrant/bm25` via fastembed) pour la composante sparse de la recherche hybride.

Les deux wrappers vivent dans `src/ragtime/embeddings/` :

- `dense.py` — `DenseEmbedder` wrappant `fastembed.TextEmbedding(intfloat/multilingual-e5-large)`
- `sparse.py` — `SparseEmbedder` wrappant `fastembed.SparseTextEmbedding(Qdrant/bm25)`

> Note : nommage des fichiers — CLAUDE.md §3 prévoit `embeddings/bge_m3.py`. On préfère `dense.py` + `sparse.py` qui décrivent fidèlement le code. Cette déviation est tracée ici (et nulle part ailleurs).

## Conséquences

### Caractéristiques techniques retenues

| | E5-large (retenu) | BGE-M3 (initialement prévu) |
|---|---|---|
| Dimensions dense | 1024 | 1024 |
| Multilingue | ~100 langues | ~100 langues |
| Sparse natif | non (BM25 complémentaire) | oui (lexical weights appris) |
| Backend | ONNX Runtime | PyTorch / ONNX (FlagEmbedding) |
| Taille modèle | 2.24 Go | 2.3 Go |
| Empreinte d'install | ~50 Mo de wheels | ~3 Go (torch+transformers) |
| Inférence CPU | ~150-300 docs/s | ~50-100 docs/s |
| MTEB MIRACL (multilingue) | ~69 % | ~70 % |

### Points d'attention pour l'implémentation

1. **Prefix obligatoire E5** : les modèles E5 attendent un préfixe selon le mode d'usage :
    - Indexation : `"passage: <texte>"`
    - Requête : `"query: <texte>"`
   Le wrapper `DenseEmbedder` doit exposer deux méthodes `embed_documents()` et `embed_queries()` qui appliquent le bon préfixe automatiquement.

2. **Le sparse n'est plus "appris"** : BGE-M3 produit un sparse sémantique (lexical weights apprises lors du training), alors que BM25 est statistique. Pour une recherche hybride sur des tickets de support courts, BM25 reste très performant (capture les termes rares, robuste sur les langues sans entraînement). À surveiller en évaluation.

3. **Reranker conservé** : `BAAI/bge-reranker-v2-m3` (CLAUDE.md §2) reste pertinent et utilisable indépendamment du modèle dense.

### Mise à jour de la stack effective

- `requirements.txt` : déjà compatible (fastembed était dans la liste).
- `.env.example` : `EMBEDDING_MODEL` passe de `BAAI/bge-m3` à `intfloat/multilingual-e5-large`.
- `src/ragtime/embeddings/__init__.py` : exporte `DenseEmbedder` et `SparseEmbedder`.

### Réversibilité

Si à terme on rebascule sur BGE-M3 (FlagEmbedding) :
- Le wrapper dense est isolé derrière `DenseEmbedder.embed_documents()` / `embed_queries()`.
- Le wrapper sparse est isolé derrière `SparseEmbedder.embed()`.
- Aucun appelant n'a connaissance du modèle sous-jacent.
- La migration consiste à remplacer le contenu de `dense.py` / `sparse.py` sans toucher au reste du pipeline.

## Alternatives écartées

- **`sentence-transformers/paraphrase-multilingual-mpnet-base-v2`** (768 dim, 50 langues) : trop éloigné de BGE-M3 en qualité MTEB.
- **`jinaai/jina-embeddings-v3`** (1024 dim, multilingue) : qualité comparable mais nécessite des wheels qui ne sont pas garanties stables sur tous OS, et expérimental pour les usages on-prem.
- **FlagEmbedding (BGE-M3 officiel)** : techniquement faisable mais coût d'installation et de runtime disproportionnés pour le MVP. À considérer pour une éventuelle V2.
