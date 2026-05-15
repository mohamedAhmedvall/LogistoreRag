# Note de veille technologique — Systèmes RAG

**Projet RAG-time / LogiStore — M2 IA & Data Science**
**Auteur :** Mohamed AHMEDVALL
**Date :** Mai 2026

---

## Préambule

Cette note de veille couvre les fondements techniques et l'état de l'art des systèmes de Retrieval-Augmented Generation (RAG), en vue de la conception d'un moteur de recherche d'entreprise pour LogiStore. Trois axes sont traités successivement : l'indexation sémantique par plongement, les variantes architecturales du RAG, et les méthodologies d'évaluation. Une synthèse finale justifie les choix techniques retenus pour le MVP.

L'objectif n'est pas de produire un état de l'art académique exhaustif mais une veille opérationnelle, orientée décision, qui doit permettre de poser les fondations de l'architecture cible et de défendre les arbitrages technologiques face à un commanditaire.

---

## Partie A — Indexation sémantique par plongement

### A.1 Du lexical au sémantique

Les moteurs de recherche traditionnels reposent sur des modèles probabilistes lexicaux comme BM25 ou TF-IDF. Ces modèles évaluent la pertinence d'un document vis-à-vis d'une requête en fonction de la fréquence d'apparition des termes et de leur rareté dans le corpus. Ils restent extrêmement performants pour des requêtes contenant des termes précis, des codes produits ou des identifiants techniques, et offrent une explicabilité immédiate : on sait exactement pourquoi un document a été retourné.

Leur limite fondamentale est l'absence de compréhension sémantique. Une requête « problème de connexion réseau » ne retournera pas un ticket intitulé « impossible d'accéder à internet » si aucun terme n'est partagé. Cette limite a motivé le développement de représentations vectorielles capables d'encoder le sens.

L'hypothèse distributionnelle, formulée dès les années 1950 (Harris, Firth), postule que deux mots apparaissant dans des contextes similaires ont des significations similaires. C'est cette hypothèse qui sous-tend l'ensemble des modèles modernes d'embeddings : un texte est projeté dans un espace vectoriel dense où la proximité géométrique reflète la proximité sémantique.

### A.2 Mesures de similarité

Trois mesures principales sont utilisées dans les espaces vectoriels d'embeddings :

La **similarité cosinus** mesure l'angle entre deux vecteurs. Comprise entre -1 et 1, elle est invariante à la magnitude, ce qui est essentiel car la norme d'un vecteur d'embedding n'a pas de signification sémantique fiable. C'est la mesure standard pour la recherche sémantique.

Le **produit scalaire** (dot product) combine direction et magnitude. Il est utilisé lorsque les vecteurs sont normalisés (cas de OpenAI text-embedding-3 par exemple), auquel cas il devient équivalent au cosinus avec un calcul légèrement plus rapide.

La **distance euclidienne** mesure une distance géométrique brute. Elle est rarement utilisée en NLP car sensible à la magnitude, mais reste pertinente pour des données déjà normalisées ou des espaces non-textuels.

Pour LogiStore, la similarité cosinus est le choix par défaut, conforme à la pratique industrielle dominante.

### A.3 Évolution des modèles d'embeddings

L'histoire des embeddings textuels suit une trajectoire claire :

**2013-2017 : embeddings statiques.** Word2Vec (Mikolov, Google), GloVe (Stanford) et FastText (Facebook) produisent un vecteur fixe par mot, indépendant du contexte. « Banque » obtient le même vecteur dans « banque financière » et « banque de sable ».

**2018-2019 : embeddings contextuels.** BERT (Devlin et al.) et ses dérivés produisent un vecteur par token *en fonction du contexte*, résolvant l'ambiguïté lexicale. Mais BERT brut n'est pas optimisé pour la similarité de phrases.

**2019-2020 : Sentence-BERT (Reimers & Gurevych).** Fine-tuning de BERT avec une architecture siamoise pour produire des embeddings de phrase comparables par cosinus. C'est l'ancêtre direct des modèles modernes de retrieval.

**2022-2024 : génération moderne.** Modèles contrastifs entraînés sur des paires (requête, document pertinent) à grande échelle. Apparition des architectures multi-représentation (dense + sparse + multi-vector dans un seul modèle).

### A.4 Bi-encoder vs cross-encoder

Cette distinction est cruciale pour comprendre l'architecture d'un système de retrieval :

Un **bi-encoder** encode la requête et le document indépendamment, en deux passes parallèles. Les embeddings produits sont comparés par une mesure de similarité simple. Avantage majeur : les embeddings des documents peuvent être pré-calculés et indexés. Inconvénient : la comparaison ne capture pas les interactions fines entre tokens de la requête et du document.

Un **cross-encoder** prend en entrée la concaténation de la requête et du document et produit un score de pertinence direct. Beaucoup plus précis car chaque token de la requête peut « attendre » sur chaque token du document. Mais le coût est rédhibitoire pour un corpus de millions de documents : il faudrait re-calculer le score pour chaque paire requête/document à chaque recherche.

La pratique industrielle combine les deux : un bi-encoder fait une présélection rapide (top-100 ou top-200), puis un cross-encoder rerank cette présélection pour produire le top-K final. C'est le pattern **retrieve-then-rerank**.

### A.5 Modèles d'embeddings actuels

Le paysage 2024-2026 est dominé par quelques familles :

**OpenAI text-embedding-3 (small, large).** Performance solide, multilingue, intégration triviale via API. Le modèle « small » offre un excellent rapport coût/performance. Inconvénient : appel API externe (problématique RGPD, latence, coûts récurrents, dépendance fournisseur).

**BGE-M3 (BAAI, 2024).** Modèle phare de la Beijing Academy of AI. Multilingue (100+ langues dont français et arabe), longueur de contexte jusqu'à 8192 tokens. Particularité unique : produit dans un même forward pass trois types de représentations : dense (embedding classique), sparse (vecteur creux type BM25 appris), et multi-vector (style ColBERT, un vecteur par token). Cette propriété permet de faire du retrieval hybride avec un seul modèle. Open-source, exécutable localement.

**Famille E5 (Microsoft).** Modèles instructifs (la requête doit être préfixée par « query: »). Versions multilingual-e5-large et multilingual-e5-base très utilisées. Bonne couverture du français.

**Nomic Embed (Nomic AI).** Premier modèle entièrement reproductible (données, code, poids), 8192 tokens, performances compétitives, licence Apache 2.0.

**mxbai-embed-large (Mixedbread).** Modèle compact (~335M paramètres) avec de très bonnes performances en anglais.

**jina-embeddings-v3.** Modèle dédié au retrieval multilingue, support LoRA pour adaptation à des domaines spécifiques.

Le **MTEB leaderboard** (Massive Text Embedding Benchmark) maintenu par Hugging Face est la référence pour comparer ces modèles sur 56 tâches de retrieval, classification et clustering. À considérer avec nuance : les performances générales ne garantissent pas la pertinence sur un domaine métier précis comme le support client en français.

### A.6 Indexation et recherche approximative (ANN)

Lorsqu'on stocke des millions de vecteurs en dimension 768 ou 1024, la comparaison exhaustive d'une requête à tous les documents devient impraticable. La complexité brute O(N·d) atteint ses limites bien avant le million de documents. Les algorithmes d'**Approximate Nearest Neighbor (ANN)** acceptent une légère perte de précision en échange d'une accélération considérable.

**HNSW (Hierarchical Navigable Small Worlds).** L'algorithme dominant en 2025. Construit un graphe en couches où les nœuds représentent les vecteurs et les arêtes connectent les voisins. La recherche descend la hiérarchie en suivant les arêtes les plus prometteuses. Paramètres clés : `M` (nombre de connexions par nœud), `ef_construction` (qualité de construction du graphe), `ef_search` (qualité de la recherche). Excellent compromis recall/latence, mais consomme de la mémoire (le graphe doit tenir en RAM). Implémenté nativement dans Qdrant, OpenSearch, Weaviate, Milvus.

**IVF (Inverted File Index).** Partitionne l'espace par k-means, puis ne fouille que les clusters les plus proches de la requête. Plus économe en mémoire que HNSW, légèrement moins précis pour un même budget de latence.

**PQ (Product Quantization).** Technique de compression qui découpe chaque vecteur en sous-vecteurs et les quantifie séparément. Permet de réduire l'empreinte mémoire d'un facteur 10 à 100 au prix d'une légère dégradation. Souvent combiné avec HNSW (HNSW-PQ) ou IVF (IVF-PQ).

**Scalar Quantization.** Compression vectorielle moins agressive (par exemple int8 au lieu de float32), divisant la mémoire par 4 sans perte significative de recall. Supportée par Qdrant via la configuration `quantization_config`.

Pour un MVP sur 10 000 à 100 000 tickets, HNSW en float32 sans quantization est largement suffisant et offre le meilleur recall. Les optimisations mémoire deviennent nécessaires au-delà de 10 millions de vecteurs.

### A.7 Panorama des bases vectorielles

Le marché des vector stores a explosé depuis 2022. Tableau comparatif synthétique :

**Qdrant.** Écrit en Rust, performance native, HNSW avec quantization fine, filtres metadata avancés (payload-aware indexing : les filtres ne se font pas en post-traitement mais sont intégrés à la recherche), support natif des vecteurs sparse, Query API hybride avec RRF intégré. Auto-hébergeable, cluster Kubernetes possible, mode embedded pour développement. Open-source Apache 2.0.

**Weaviate.** Plus orienté « knowledge graph » avec son schéma typé. API GraphQL et REST. Modules d'embeddings et de génération intégrés (peut appeler OpenAI directement). Multi-modalité.

**Milvus / Zilliz Cloud.** Architecture distribuée native, excellente scalabilité (milliards de vecteurs), support GPU. Interface graphique Attu. Plus complexe à opérer en self-hosted.

**Pinecone.** SaaS uniquement, simplicité d'API maximale, performance solide. Coût élevé en production, pas d'option on-premise (rédhibitoire pour des secteurs régulés).

**OpenSearch / Elasticsearch.** Plateforme de search mature avec ajout de k-NN. Avantage : BM25 et vecteurs dans un même moteur, écosystème enterprise (Kibana, ingestion, sécurité fine). Inconvénient : empreinte JVM, complexité opérationnelle, moins optimisé que les vector-natives sur la partie vectorielle pure.

**pgvector.** Extension PostgreSQL. Simplicité maximale : si l'organisation a déjà du Postgres, ajouter une recherche vectorielle ne demande qu'une extension. Mais hybride manuel (BM25 doit être assemblé à part), scalabilité limitée au-delà de quelques millions de vecteurs.

**ChromaDB.** Très simple, embedded, idéal pour le développement et les prototypes. Moins adapté à la production.

**FAISS (Facebook AI Similarity Search).** Bibliothèque (pas une base de données). Extrêmement rapide, mais ne gère ni la persistence native ni les filtres metadata. Souvent utilisé comme brique sous-jacente par des solutions de plus haut niveau.

### A.8 Synthèse partie A

Pour LogiStore, le choix retenu est **Qdrant + BGE-M3**. Cette combinaison offre :
- Une recherche hybride dense + sparse native, sans assemblage manuel
- Un modèle d'embeddings multilingue capable de gérer le français des tickets
- Une indépendance totale vis-à-vis des API externes (RGPD-friendly)
- Une scalabilité éprouvée jusqu'à des centaines de millions de vecteurs
- Une explicabilité fine grâce aux filtres metadata payload-aware

---

## Partie B — Variantes architecturales du RAG

### B.1 Naive RAG (RAG 1.0)

Le pattern originel, introduit par Lewis et al. en 2020, suit un pipeline linéaire en cinq étapes : segmentation du corpus en chunks, encodage des chunks en vecteurs, stockage dans une base vectorielle, recherche des top-K chunks similaires à la requête, injection dans le prompt d'un LLM générateur.

Ce pattern est extrêmement simple à implémenter et donne des résultats étonnamment corrects sur des cas d'usage bien cadrés. Il souffre cependant de plusieurs limites structurelles :

La **segmentation arbitraire** des documents en chunks de taille fixe coupe souvent au milieu de phrases ou de raisonnements, perdant le contexte nécessaire à la compréhension. Un ticket de support et sa résolution peuvent se retrouver dans des chunks distincts, non co-récupérés.

Le **retrieval purement sémantique** peut manquer des matches lexicaux évidents. Une requête contenant un code produit précis (« erreur ERR-4421 ») peut retourner des résultats sémantiquement « proches » mais ne contenant pas ce code exact.

L'**absence de filtrage qualité** signifie que des chunks peu pertinents mais sémantiquement bruyants peuvent polluer le contexte du LLM, déclenchant des hallucinations ou des réponses confuses.

L'**absence de validation** de la réponse générée laisse passer des hallucinations même lorsque le contexte fourni est correct.

### B.2 Advanced RAG

Le terme « Advanced RAG » regroupe un ensemble de techniques qui interviennent en amont (pré-retrieval) ou en aval (post-retrieval) du retrieval principal.

**Techniques de pré-retrieval :**

*Query rewriting.* Un LLM reformule la requête utilisateur pour la rendre plus adaptée au retrieval. Par exemple, transformer « ça marche pas le truc d'hier » en « erreur survenue lors de la session du 12 mai ». Utile dans un contexte conversationnel.

*Query expansion.* Génération de variations ou synonymes de la requête, puis fusion des résultats. Particulièrement utile sur les requêtes courtes.

*HyDE (Hypothetical Document Embeddings).* Idée contre-intuitive : on demande à un LLM de générer un document hypothétique qui répondrait à la question, on encode ce document, et on cherche par similarité avec lui. Cela rapproche la requête (souvent courte) de la distribution des documents (souvent longs) dans l'espace d'embedding.

*Multi-query.* Génération de N reformulations parallèles de la requête, retrieval pour chacune, fusion des résultats par RRF.

*Step-back prompting.* Reformulation par abstraction : « comment résoudre l'erreur X sur produit Y » devient « quels sont les principes de diagnostic des erreurs sur produit Y », élargissant la recherche.

*Routing.* Un classifieur ou un LLM décide à quelle source de données adresser la requête (tickets ? procédures ? FAQ ?). Permet de spécialiser les retrievers.

**Techniques de post-retrieval :**

*Reranking par cross-encoder.* Comme évoqué en partie A, un cross-encoder rerank le top-K du bi-encoder. Modèles populaires : BGE-reranker-v2-m3 (multilingue, BAAI), ms-marco-MiniLM (anglais, léger), Cohere Rerank (SaaS, performant).

*Contextual compression.* Un LLM filtre ou résume les chunks récupérés pour ne garder que les portions vraiment pertinentes à la question, économisant de la fenêtre de contexte et réduisant le bruit.

*Sentence-window retrieval.* Indexer des phrases courtes (pour la précision sémantique) mais retourner la phrase + ses voisines (pour le contexte). Implémenté nativement dans LlamaIndex.

*Parent-document retrieval.* Indexer des chunks fins mais retourner le document parent. Variante du précédent.

### B.3 Modular RAG

À mesure que les techniques se sont multipliées, une vision « modulaire » du RAG s'est imposée. Plutôt que de penser le pipeline comme une séquence figée, on le conçoit comme un assemblage de briques recombinables : retrievers, rerankers, routeurs, générateurs, validateurs. Cette modularité permet d'adapter l'architecture aux caractéristiques du domaine et de la requête.

Le pattern **Rewrite-Retrieve-Read** illustre cette modularité : une requête utilisateur est d'abord réécrite par un LLM, puis utilisée pour le retrieval, puis le contexte récupéré sert à la génération finale. Trois briques séparées et remplaçables indépendamment.

Le pattern **ITER-RETGEN** (iterative retrieval and generation) introduit une boucle : une première génération est utilisée pour faire un nouveau retrieval, qui alimente une nouvelle génération, et ainsi de suite jusqu'à convergence. Utile pour les questions multi-hop.

### B.4 Hybrid Search (BM25 + vecteurs)

C'est probablement la technique la plus impactante en termes de qualité, pour un effort d'implémentation modéré. L'idée : combiner la précision lexicale de BM25 avec la flexibilité sémantique des embeddings.

**Pourquoi la fusion fonctionne.** BM25 et les embeddings ont des points forts complémentaires. BM25 excelle sur les requêtes contenant des termes précis (codes, noms propres, identifiants), où il offre une garantie de match exact. Les embeddings excellent sur les requêtes paraphrasées, abstraites, ou utilisant un vocabulaire différent de celui des documents. Combinés, les deux signaux compensent leurs faiblesses respectives.

**Méthodes de fusion :**

*Reciprocal Rank Fusion (RRF).* Méthode la plus utilisée. Pour chaque document, on calcule un score basé non sur les scores bruts de chaque retriever (qui ne sont pas comparables) mais sur les rangs. Formule : `score(d) = Σ_i 1 / (k + rank_i(d))`, avec k=60 par convention. Insensible aux différences d'échelle entre BM25 et cosinus. Excellent comportement empirique. Implémenté nativement dans Qdrant, OpenSearch, et la plupart des frameworks.

*Linear weighted fusion.* `score(d) = α · score_dense(d) + (1-α) · score_sparse(d)`. Nécessite une normalisation préalable des scores (min-max, z-score), sinon les échelles dominent. Plus simple à interpréter mais plus fragile.

*Distribution-Based Score Fusion (DBSF).* Méthode plus robuste qui modélise les distributions de scores par retriever et fait une fusion bayésienne. Supportée par Qdrant.

**Implémentations modernes :**

Qdrant propose depuis 2024 une **Query API hybride** qui prend en entrée plusieurs requêtes (dense, sparse, multi-vector) et applique une stratégie de fusion (RRF ou DBSF) en une seule requête. Cela évite l'aller-retour client pour fusionner les résultats.

OpenSearch propose la **hybrid query** combinant BM25 et k-NN avec normalisation et pondération configurables.

BGE-M3 est particulièrement intéressant pour le retrieval hybride car il produit dense + sparse en un seul appel : on n'a pas à maintenir un modèle BM25 séparé.

### B.5 GraphRAG

Microsoft a publié en 2024 GraphRAG, qui introduit un paradigme différent : indexer le corpus non pas comme une collection de chunks mais comme un graphe de connaissances. Un LLM extrait les entités et leurs relations à partir du corpus, construit le graphe, puis résume hiérarchiquement les communautés détectées par algorithmes de clustering de graphe (Leiden).

Au moment de la requête, deux modes : **local search** (similaire au RAG vectoriel classique mais enrichi par le graphe) et **global search** (interroge les résumés de communautés pour répondre à des questions globales sur le corpus, comme « quels sont les thèmes principaux ? »).

GraphRAG brille sur les requêtes globales ou multi-hop, là où le RAG vectoriel classique échoue. Le coût d'indexation est cependant élevé (de nombreux appels LLM lors de l'ingestion) et la complexité d'implémentation considérable.

Pour LogiStore, GraphRAG est pertinent à moyen terme pour des cas d'usage comme « quelles sont les tendances de tickets sur le produit X ce trimestre ? », mais hors-périmètre du MVP.

### B.6 Agentic RAG

Dans un agentic RAG, le LLM ne suit pas un pipeline figé : il **décide** quand faire du retrieval, quelle source interroger, quelles requêtes formuler, et quand s'arrêter. Cette approche utilise le **tool use** des LLM modernes : le LLM dispose d'outils (search_tickets, search_products, query_database) et choisit lui-même lesquels appeler.

L'agentic RAG permet de gérer naturellement les questions multi-hop, les négations, les comparaisons inter-sources. Inconvénients : latence (plusieurs allers-retours LLM par requête), coût, débuggabilité plus difficile, comportements émergents parfois imprévisibles.

Frameworks populaires : LangGraph (orchestration de graphes d'agents), LlamaIndex (composants agentic intégrés), CrewAI (multi-agents).

### B.7 Self-RAG et Corrective RAG (CRAG)

Deux variantes récentes qui visent à améliorer la fiabilité du RAG.

**Self-RAG** (Asai et al., 2023) entraîne un LLM à émettre des tokens de réflexion spéciaux : `[Retrieve]` pour décider s'il faut faire un retrieval, `[Relevance]` pour évaluer la pertinence des passages récupérés, `[Support]` pour vérifier que la réponse est ancrée dans le contexte. Le modèle apprend à s'auto-critiquer.

**Corrective RAG (CRAG)** (Yan et al., 2024) introduit un évaluateur léger qui score la qualité du retrieval. Si la qualité est insuffisante, le système déclenche une recherche web (ou une autre source) en fallback. Approche pragmatique qui ne nécessite pas de fine-tuning du LLM principal.

### B.8 Synthèse partie B

Pour LogiStore, l'architecture cible combine plusieurs niveaux :

- **MVP (court terme) :** RAG hybride dense + sparse, avec RRF natif Qdrant. Reranking optionnel avec BGE-reranker. Pas de pré-retrieval avancé, pas d'agentique.
- **Évolution moyen terme :** ajout de query rewriting et HyDE pour gérer les requêtes courtes du support ; reranking systématique ; routing entre sources (tickets/produits/procédures).
- **Long terme :** agentic RAG pour les usages analytiques (« quels sont les motifs récurrents sur produit X ? ») ; GraphRAG pour les vues transverses du corpus support.

---

## Partie C — Évaluation des systèmes RAG

L'évaluation d'un RAG est notoirement difficile car le système combine deux composants évaluables séparément (retrieval et génération) plus une dynamique d'interaction. Une bonne stratégie d'évaluation distingue ces niveaux.

### C.1 Métriques de retrieval (information retrieval classique)

Ces métriques sont indépendantes du LLM générateur et évaluent uniquement la qualité du retriever. Elles supposent l'existence d'un **golden dataset** : pour chaque requête, on connaît la liste des documents pertinents.

**Precision@K.** Proportion de documents pertinents dans les K premiers résultats. Mesure la « pureté » du top-K. Si K=10 et que 7 résultats sont pertinents, Precision@10 = 0.7.

**Recall@K.** Proportion des documents pertinents qui apparaissent dans le top-K. Mesure la « couverture ». Si 5 documents sont pertinents dans le corpus et que 4 apparaissent dans le top-10, Recall@10 = 0.8.

**Mean Reciprocal Rank (MRR).** Pour chaque requête, on prend l'inverse du rang du premier document pertinent. On moyenne sur toutes les requêtes. Pénalise lourdement les retrievers qui mettent les bons résultats en bas de liste. Adapté aux usages où l'utilisateur ne regarde que le premier résultat.

**Normalized Discounted Cumulative Gain (NDCG@K).** Métrique plus sophistiquée qui prend en compte à la fois la position (les premiers résultats comptent plus) et le degré de pertinence (un document « très pertinent » vaut plus qu'un document « moyennement pertinent »). C'est la métrique de référence dans les benchmarks de search (TREC, MS MARCO).

**Hit Rate@K.** Binaire : 1 si au moins un document pertinent apparaît dans le top-K, 0 sinon. Métrique « grosse maille » utile en feedback rapide.

Pour LogiStore, je retiens Precision@5, Recall@10, NDCG@10 et MRR. Ces métriques se complètent : Precision@5 mesure ce que l'utilisateur voit en premier, NDCG@10 mesure le classement global, MRR mesure la rapidité d'accès au premier résultat utile.

### C.2 Le RAG Triad

Quand on évalue le système complet (retrieval + génération), TruLens a popularisé le cadre dit du **RAG Triad** :

**Context Relevance.** Les passages récupérés sont-ils pertinents pour la question posée ? Cette métrique évalue le retrieval, mais d'un point de vue « apparent » : sans golden dataset, on demande à un juge (humain ou LLM) si chaque passage retourné aide à répondre à la question.

**Groundedness (ou Faithfulness).** La réponse générée est-elle ancrée dans le contexte fourni ? C'est la métrique anti-hallucination par excellence. On vérifie si chaque affirmation de la réponse peut être justifiée par un passage du contexte.

**Answer Relevance.** La réponse répond-elle effectivement à la question posée ? Une réponse peut être ancrée et vraie mais hors-sujet.

Les trois métriques sont indispensables : un système qui retourne du bon contexte mais hallucine échoue sur la groundedness ; un système qui répond fidèlement mais à côté de la plaque échoue sur l'answer relevance.

### C.3 Métriques avancées (RAGAS)

Le framework **RAGAS** (RAG Assessment) propose une boîte à outils Python avec des métriques calculables automatiquement à partir d'un LLM juge :

- **Context Precision** et **Context Recall** : précision et rappel du contexte par rapport à une vérité terrain.
- **Faithfulness** : décomposition de la réponse en affirmations atomiques, vérification de chaque affirmation contre le contexte.
- **Answer Relevancy** : génération de questions à partir de la réponse, comparaison de leur embedding à la question originale.
- **Answer Semantic Similarity** : similarité sémantique réponse / réponse de référence.
- **Answer Correctness** : combinaison de similarité sémantique et de correspondance factuelle.

RAGAS est devenu un standard de facto en 2024-2025. Limite : la fiabilité des métriques dépend entièrement de la fiabilité du LLM juge.

### C.4 LLM-as-judge

L'évaluation manuelle par expert métier est l'étalon-or mais ne passe pas à l'échelle. Le LLM-as-judge consiste à utiliser un LLM puissant (typiquement GPT-4, Claude Opus ou équivalent) comme évaluateur automatique. La méthode est devenue centrale dans l'évaluation des systèmes génératifs.

**Principes de prompting d'un juge :**

Un bon prompt de juge est structuré : il explique le critère à évaluer, donne l'échelle de notation (binaire, Likert 1-5, score continu), demande une justification (chain-of-thought) avant la note, et fournit éventuellement des exemples calibrés (few-shot). Le format de sortie est strict (JSON parsable de préférence).

**Biais documentés du LLM-as-judge :**

*Position bias.* En évaluation pairwise (deux réponses comparées), le LLM tend à préférer la réponse présentée en premier. Mitigation : randomiser l'ordre, ou tester les deux ordres et moyenner.

*Length bias.* Les réponses longues sont systématiquement favorisées, indépendamment de leur qualité. Mitigation : pénaliser explicitement la longueur dans le prompt, ou normaliser.

*Self-preference.* GPT préfère les réponses de GPT, Claude préfère les réponses de Claude. Critique en évaluation comparative inter-modèles.

*Verbosity bias.* Préférence pour les réponses avec beaucoup de structure visuelle (listes, headers), indépendamment du contenu.

*Anchoring.* Les exemples fournis dans le prompt ancrent fortement les notes.

**Mitigations :**

- Multi-juge avec agrégation (vote majoritaire, moyenne)
- Calibration sur un sous-ensemble annoté humainement
- Position swapping systématique
- Décomposition en sous-critères (au lieu d'un score global)

### C.5 Frameworks d'évaluation

**RAGAS** (Python, open-source). Le plus populaire. Métriques out-of-the-box, intégration avec LangChain et LlamaIndex.

**TruLens** (open-source). Instrumentation et tracking, dashboard de visualisation, RAG Triad natif. Très pratique pour le suivi en production.

**DeepEval** (open-source). Approche « tests unitaires pour LLM » à la pytest. Pertinent pour intégrer l'évaluation en CI/CD.

**ARES** (Stanford, recherche). Framework académique qui fine-tune un juge sur un domaine spécifique pour améliorer la fiabilité.

**Open RAG Benchmark (Vectara).** Benchmark reproductible centré sur les PDF multimodaux.

### C.6 Construction d'un golden dataset

L'évaluation rigoureuse nécessite un jeu de test métier. Trois approches :

**Manuel.** Un expert métier rédige les questions et identifie les passages attendus. Très qualitatif mais coûteux. Indispensable pour calibrer les méthodes automatiques.

**Synthétique.** Un LLM génère des questions à partir des chunks du corpus. RAGAS propose un `TestsetGenerator` qui produit différents types de questions (simples, multi-hop, conditionnelles). Permet d'obtenir rapidement un volume important, au prix d'un biais : les questions générées par LLM ne ressemblent pas toujours aux vraies questions des utilisateurs.

**Hybride.** Génération synthétique massive, puis validation et correction humaine sur un échantillon. Meilleur compromis volume/qualité.

**Couverture du dataset.** Un bon golden dataset couvre plusieurs types de requêtes : factuelles simples (« qui est responsable du ticket X »), multi-hop (« quel a été le délai moyen de résolution pour les tickets de catégorie Y au T1 »), avec négation (« quels tickets ne sont pas résolus »), comparatives, hors-sujet (pour tester les refus). Pour LogiStore, je prévois 30 à 50 requêtes couvrant ces types.

### C.7 Synthèse partie C

Pour LogiStore, la stratégie d'évaluation retenue est multi-niveaux :

- **Évaluation retrieval :** golden dataset manuel de 30 requêtes annotées, métriques Precision@5, Recall@10, NDCG@10, MRR.
- **Évaluation génération :** RAG Triad via LLM-as-judge (Claude ou Mistral via OpenRouter), avec prompt structuré et position swapping.
- **Métriques automatisées :** RAGAS pour faithfulness et answer relevance.
- **Validation humaine :** annotation par un référent métier sur un sous-ensemble de 10 requêtes pour calibrer le juge.

---

## Synthèse et choix technologiques pour LogiStore

Sur la base de cette veille, l'architecture retenue pour le MVP repose sur :

**Indexation.** Qdrant en self-hosted (Docker) avec HNSW en float32. Cohabitation de vecteurs dense (BGE-M3, 1024 dim) et sparse (BM25 généré par BGE-M3 ou fastembed). Métadonnées indexées : `category`, `product`, `customer_id`, `status`, `created_at` pour filtrage payload-aware.

**Modèle d'embeddings.** BGE-M3, exécuté localement. Justifications : multilingue (couvre les futurs corpus francophones et internationaux), génère dense et sparse en un forward pass (économie d'infrastructure), licence permissive, indépendance d'API externe (RGPD).

**Recherche.** Query API hybride de Qdrant avec RRF. Reranking optionnel via BGE-reranker-v2-m3 pour les requêtes critiques.

**Génération.** LLM accédé via OpenRouter (Claude Sonnet ou Mistral selon le cas d'usage), uniquement pour la synthèse optionnelle des top-K résultats. Pas de génération sans contexte récupéré, pas d'agentique en MVP.

**Évaluation.** RAGAS pour les métriques automatisées, jeu de test métier de 30 requêtes annotées manuellement, LLM-as-judge custom avec position swapping.

**Trajectoire d'évolution.** L'architecture est compatible avec une montée en gamme progressive : ajout de query rewriting, reranking systématique, routing multi-sources, puis à plus long terme GraphRAG pour les usages analytiques et agentic RAG pour les workflows complexes.

---

## Références

### Architectures RAG

1. Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, 2020. https://arxiv.org/abs/2005.11401
2. Gao et al., *Retrieval-Augmented Generation for Large Language Models: A Survey*, 2024. https://arxiv.org/abs/2312.10997
3. Zhao et al., *Retrieval-Augmented Generation: A Comprehensive Survey*, 2025. https://arxiv.org/abs/2506.00054
4. Fan et al., *A Survey on RAG Meeting LLMs*, 2024. https://arxiv.org/abs/2405.06211
5. RAG-Survey (repo). https://github.com/hymie122/RAG-Survey

### Variantes avancées

6. Asai et al., *Self-RAG*, 2023. https://arxiv.org/abs/2310.11511
7. Yan et al., *Corrective RAG (CRAG)*, 2024. https://arxiv.org/abs/2401.15884
8. Microsoft, *GraphRAG*. https://github.com/microsoft/graphrag
9. Gao et al., *Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)*, 2022. https://arxiv.org/abs/2212.10496

### Embeddings et retrieval

10. Reimers & Gurevych, *Sentence-BERT*, 2019. https://arxiv.org/abs/1908.10084
11. Chen et al., *BGE-M3: Multi-Functionality, Multi-Linguality, Multi-Granularity*, 2024. https://arxiv.org/abs/2402.03216
12. Wang et al., *Multilingual E5 Text Embeddings*, 2024. https://arxiv.org/abs/2402.05672
13. MTEB Leaderboard. https://huggingface.co/spaces/mteb/leaderboard

### Indexation vectorielle

14. Malkov & Yashunin, *Efficient and robust approximate nearest neighbor search using HNSW graphs*, 2018. https://arxiv.org/abs/1603.09320
15. Documentation Qdrant. https://qdrant.tech/documentation/
16. OpenSearch Vector Search. https://docs.opensearch.org/latest/vector-search/

### Évaluation

17. Es et al., *RAGAS: Automated Evaluation of RAG*, 2023. https://arxiv.org/abs/2309.15217
18. Saad-Falcon et al., *ARES: Automated Evaluation Framework for RAG*, 2023. https://arxiv.org/abs/2311.09476
19. Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*, 2023. https://arxiv.org/abs/2306.05685
20. Mistral AI, *Evaluating RAG with LLM as a Judge*. https://mistral.ai/news/llm-as-rag-judge
21. Open RAG Benchmark (Vectara). https://github.com/vectara/open-rag-bench

### Dataset

22. Customer Support Tickets Dataset (200K+ Records), Kaggle. https://www.kaggle.com/datasets/mirzayasirabdullah07/customer-support-tickets-dataset-200k-records
