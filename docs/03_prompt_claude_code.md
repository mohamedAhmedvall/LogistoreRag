# Prompt Claude Code — Développement MVP RAG-time / LogiStore

> Ce document est le **brief initial** à transmettre à Claude Code lors du démarrage du développement. Il complète `CLAUDE.md` (qui contient les règles permanentes) en définissant les objectifs et le déroulé du chantier.
>
> **Usage :** copier-coller la section "PROMPT À ENVOYER" ci-dessous dans Claude Code après avoir cloné/initialisé le repo et placé `CLAUDE.md`, `README.md`, `pyproject.toml` et les autres fichiers fournis à la racine.

---

## PROMPT À ENVOYER (copier-coller intégralement)

```
Bonjour. Tu vas développer le MVP du projet RAG-time / LogiStore avec moi.

## Contexte

Tout le contexte permanent du projet est dans CLAUDE.md à la racine du repo. Lis-le intégralement avant toute action — il définit la mission, la stack technique imposée, l'arborescence à respecter, les conventions de code, les contraintes critiques, et l'ordre de développement recommandé.

Lis également :
- README.md pour la vue utilisateur
- docs/01_veille_technologique.md pour le fondement technique
- docs/02_architecture.md pour la cartographie cible
- .env.example pour les variables de configuration

## Objectif de cette session

Développer un MVP complet, fonctionnel et testé, du moteur de recherche RAG hybride sur les tickets de support LogiStore. Le MVP doit passer la "Définition du done" décrite en section 10 de CLAUDE.md.

## Méthode de travail

Tu vas procéder par incréments verticaux. Pour chaque incrément :

1. Annoncer clairement ce que tu vas faire
2. Implémenter (en suivant l'ordre de développement de CLAUDE.md §7)
3. Écrire les tests
4. Lancer les tests, montrer qu'ils passent
5. Me demander si je veux valider avant de passer à l'incrément suivant

Ne fais PAS tout en une fois. Découpe en étapes courtes pour que je puisse valider à chaque jalon.

## Incrément 1 — Fondations

Commence par poser les fondations :

1. Créer l'arborescence complète vide (selon CLAUDE.md §3), avec des `__init__.py` partout
2. Implémenter `src/ragtime/config.py` (Pydantic Settings basé sur .env.example)
3. Implémenter `src/ragtime/logging_setup.py` (structlog en JSON)
4. Implémenter `src/ragtime/models.py` avec les schémas principaux :
   - `Ticket` (modèle source)
   - `IndexableDocument` (forme gold prête à indexer)
   - `SearchQuery` (requête API)
   - `SearchResult` (résultat unitaire)
   - `SearchResponse` (réponse API complète)
   - `SynthesisRequest`, `SynthesisResponse`
5. Implémenter les exceptions custom (`exceptions.py`)
6. Créer la conftest.py minimale pour pytest
7. Lancer `make lint && make test` pour valider

À l'issue de cet incrément : tu me montres la structure créée, le résultat des lint/tests, et tu attends ma validation.

## Incréments suivants (référence — tu détailles à chaque étape)

2. Pipeline d'ingestion (loader Kaggle, normalizer, chunker, script run_ingestion)
3. Embeddings BGE-M3 (wrapper dense + sparse, batching, tests)
4. Couche d'indexation Qdrant (client, schema, indexer idempotent)
5. Ingestion end-to-end (script run_ingestion fonctionnel sur 1000 tickets pour itération)
6. Couche recherche hybride (hybrid search RRF, filtres metadata)
7. API FastAPI (routes search, health, docs OpenAPI)
8. Frontend Streamlit basique (recherche + résultats)
9. Reranking optionnel (BGE-reranker-v2-m3)
10. Couche LLM (provider abstrait, OpenRouter, synthèse avec citations)
11. Évaluation (golden dataset, métriques IR, LLM-as-judge, runner)
12. Polish (README, captures, documentation finale)

## Règles strictes

- Tu suis CLAUDE.md à la lettre, en particulier la stack et l'arborescence
- Tu n'introduis aucune dépendance hors de requirements.txt sans me demander
- Tu n'utilises jamais print() dans le code applicatif (uniquement logger)
- Tu n'écris jamais de secret en dur
- Tu écris des tests pour toute logique non-triviale
- Tu utilises des type hints partout
- Tu commits régulièrement avec des messages descriptifs (Conventional Commits : feat:, fix:, test:, docs:, refactor:)
- Si tu rencontres une ambiguïté, tu me demandes avant de décider

## Démarre par l'incrément 1.

Quand tu commences, dis-moi explicitement "Je commence l'incrément 1 — Fondations" et liste les fichiers que tu vas créer.
```

---

## Notes d'utilisation

**Quand l'envoyer.** Une fois que tu as :
- Initialisé le repo Git
- Placé `CLAUDE.md`, `README.md`, `pyproject.toml`, `requirements.txt`, `docker-compose.yml`, `Makefile`, `.env.example`, `.gitignore` à la racine
- Placé les 4 documents dans `docs/` (`01_veille_technologique.md`, `02_architecture.md`, `architecture.png`, et les 2 autres à produire ensuite)
- Activé l'environnement virtuel et installé les dépendances (`make install`)

**Comment l'envoyer.** Dans Claude Code, démarrer une nouvelle session dans le répertoire du repo, et envoyer le contenu du bloc PROMPT À ENVOYER en premier message.

**Vérifications après chaque incrément :**
- Claude Code respecte-t-il l'arborescence ? (`tree -L 3`)
- Les tests passent-ils ? (`make test`)
- Le lint passe-t-il ? (`make lint`)
- Les commits sont-ils propres ? (`git log --oneline`)

**Si Claude Code dévie.** Lui rappeler explicitement de lire CLAUDE.md. La règle d'or : on ne corrige pas en aval ce qui peut être imposé en amont.
