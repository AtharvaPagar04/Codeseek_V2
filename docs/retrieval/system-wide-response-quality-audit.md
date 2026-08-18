# CodeSeek System-Wide Response-Quality Audit

## A. Executive verdict

CodeSeek’s dominant general response-quality limitation is retrieval and evidence selection, not prose generation. The pipeline commonly loses required evidence before generation through incomplete indexability, single-intent query compression, hard top-k limits, heuristic reranking, late or disabled structural traversal, and rank-based source gating.

Generation is nevertheless the dominant limitation for one severe class: empty streamed responses. Four of twenty persisted assistant messages are empty even though retrieval traces contain usable evidence. The streaming adapter has no enforced non-empty-result invariant and the client sees streamed text before final validation.

The highest-impact confirmed mechanisms are:

1. No end-to-end representation of requested facets or evidence coverage.
2. Authoritative artifacts and relationships are excluded, weakly indexed, disabled, or introduced too late.
3. Streaming can successfully finish with no recognized text and persist an empty answer.

False absence is especially unsafe: one active route declares a “comprehensive scan” without performing repository retrieval.

---

## B. Audit scope and evidence quality

This was a read-only audit. No files, containers, branches, indexes, databases, sessions, traces, or providers were modified or invoked.

Opening and closing checks both reported:

- Branch: `develop`
- Worktree: the same 47 pre-existing modified/untracked entries
- Final status digest: `78101da8309c784a16b14672e8efc327ace886f5c444cb3fe278aac473d8dc3c`
- Audit-created worktree changes: none

Evidence used:

- Current source and effective configuration.
- Source hashes from the running backend image, matching the inspected workspace for the principal runtime files.
- Read-only PostgreSQL, Qdrant, container, and persisted-trace inspection.
- Existing fixtures, reports, and evaluation datasets.
- Isolated tests using temporary SQLite state.
- Documentation only as comparison evidence, never as proof of runtime behavior.

Available runtime evidence was narrow:

- One ready repository session.
- 27 relational `session_files`, 48 relational chunks, and 49 Qdrant points.
- 52 graph nodes and 95 edges: 48 `contains`, 15 `defines`, 32 `imports`.
- 21 persisted retrieval traces.
- 20 user and 20 assistant messages; 4 assistant messages are empty.
- The only active indexed repository was a Portfolio repository.

Therefore:

- Mechanisms visible directly in active code/configuration are **Confirmed**.
- Cross-repository effects supported by different synthetic fixtures are generally **High** or **Moderate confidence**.
- Claims requiring real Java, Go, Rust, large monorepo, or multiple live repository traces remain **Unresolved**.
- A historical CodeSeek evaluation was contaminated by Portfolio candidates. It is evidence of an isolation/evaluation-control failure, not a trustworthy current quality score.

Confidence labels below mean:

- **Confirmed:** directly established by active code or current runtime state.
- **High confidence:** code plus multiple tests/traces or independent examples.
- **Moderate confidence:** mechanism exists, but representative runtime evidence is limited.
- **Unresolved:** observability is insufficient to identify the actual cause.

---

## C. Verified current pipeline

Effective flags in the running backend were:

- Dense retrieval: enabled
- Lexical retrieval: enabled
- Scored intent: enabled
- LLM intent classifier: disabled
- Two-layer sources: enabled
- Graph shadow retrieval: disabled
- Graph active retrieval: disabled
- Graph Assist user toggle: disabled
- History injection by default: disabled
- Debug diagnostics: enabled
- Answer-trace logging: disabled

Host `.env` values suggesting graph enablement were not present in the running backend environment. The effective runtime is therefore graph-disabled.

| Stage | Active implementation | Input | Output | Limits/flags | Failure handling |
| ----- | --------------------- | ----- | ------ | ------------ | ---------------- |
| Discovery | `backend/rag_ingestion/stages/discovery.py` | Repository root | Paths | `.gitignore`, directory and symlink rules | Excluded paths disappear entirely |
| Filtering/language | `backend/rag_ingestion/stages/filtering.py`, `language.py` | Paths | Supported files | Explicit extensions and path exclusions | Unsupported/ignored artifacts are not indexed |
| Parsing/chunking | `parser.py`, `chunker.py` | File text | File and symbol chunks | AST only for Python/JS/TS families; overflow at about 2,048 tokens | Parse failure falls back to file-level text |
| Description/embedding | `embedder.py` | Chunk and metadata | 384-dimensional vector | Code prefix about 6,000 characters; input about 10,000 | Failed embeddings can become zero vectors |
| Storage/state | `storage.py`, `state.py` | Chunk/vector | Qdrant payload and incremental state | State uses path, size, mtime | Reused state may skip absent/stale index content |
| Graph build | Ingestion graph builder | Chunk metadata | `contains`, `defines`, `imports` graph | No active call/inheritance graph | Graph failure does not prevent repository readiness |
| API entry | `backend/retrieval/api_service.py` | Session/thread/query | Bound repository query | Session/collection validation | Errors become API events or failures |
| Memory/query processing | `follow_up_memory.py`, `query_processor.py` | Raw query and recent source entities | Rewritten query, primary intent, targets | Confidence floor `0.50`; deterministic classifier | Low-confidence targetless query becomes retrieval-free `OUT_OF_SCOPE` |
| Candidate retrieval | `searcher.py` | Query information | Dense, lexical, exact, metadata, heuristic candidates | 15 dense, 15 lexical, 10 post-merge | Timeouts/retries; empty set continues to low-context handling |
| Fusion/reranking | `searcher.py` | Channel candidates | Ranked top candidates | RRF `k=60`; relevance `0.45`; per-file cap 2 | Below-threshold/top-k candidates are discarded |
| Graph/expansion | `graph/retrieval.py`, `expander.py` | Surviving candidates | Related chunks | Graph off; expansion two hops, total six | Missing/unresolved relations produce no evidence |
| Assembly/gating | `assembler.py`, `source_filter.py` | Ranked/expanded candidates | Context, display sources, reasoning sources | Context 7,000; display 6; reasoning 12 | Low-ranked or late supporting evidence is removed |
| Routing/generation | `main.py`, `llm.py` | Query, context, sources | Deterministic or provider answer | Route-specific builders; 3 retries for sync | Sync has fallback text; stream only falls back on exception |
| Stream parsing | `llm.py` | Provider SSE | Text deltas | Reads `choices[0].delta.content` only | Zero recognized deltas may finish successfully |
| Validation | `answer_validation.py` | Generated text and allowed sources | Repaired answer | Path/code/numeric checks for selected modes | Generic empty answer is not rejected |
| Persistence/API | `api_service.py`, `frontend/src/utils/api.js` | Deltas, final text, sources | Stored message and client events | Sources then `done` event | Persisted answer can be empty; client can miss unterminated final buffer |

---

## D. Generalized failure-pattern matrix

Evidence abbreviations: **C** current code/config, **R** current runtime/traces, **T** tests/fixtures, **E** stored evaluations.

| Query/failure pattern | Expected invariant | Current mechanism | Common failure point | Downstream symptom | Evidence | Generality | Confidence |
| --------------------- | ------------------ | ----------------- | -------------------- | ------------------ | -------- | ---------- | ---------- |
| 1. Direct file lookup | Exact path survives ranking | Exact Qdrant plus filesystem fallback | Narrow file regex; ambiguous basename capped at 2 | Wrong file or incomplete file | C, T | Cross-language only for indexed files | High confidence |
| 2. Direct symbol lookup | Definition is returned | Exact symbol plus regex fallback | Limited parser/language set; duplicate names lack module disambiguation | Consumer or unrelated definition | C, T | Python/JS/TS strongest | High confidence |
| 3. Exact source location | Exact evidence bypasses generic thresholds | Deterministic location route | Only visible/allowed source set reaches builder | Correct location omitted or stripped | C, T | Generic after retrieval | High confidence |
| 4. Imported definition | Consumer and definition both survive | Import backing and expansion | Unsupported import forms, six-source limit, late tier | Consumer without source of truth | C, T, R | Language-dependent | Confirmed |
| 5. Caller/callee | Direction and module identity preserved | Payload `calls` plus callee lookup | Calls incomplete; global symbol lookup; “caller” query semantics inconsistent | Wrong direction or same-name target | C, T | Weak outside simple functions | High confidence |
| 6. Dependency/import trace | Resolver follows configured paths | Import resolver, re-export tracing | CommonJS/dynamic imports/nested aliases absent | Broken dependency chain | C, T | JS/TS/Python subset | Confirmed |
| 7. Multi-hop flow | Every necessary hop represented | Two-hop expansion after top-10 | First-hop evidence must already survive; six-chunk cap | First step only | C, T | Generic mechanism, narrow metadata | High confidence |
| 8. Architecture overview | Major subsystems and entry points covered | Repo summary, anchors, overview builder | Hardcoded anchor/file-role rules; summary can be stale/partial | Familiar-layout narrative | C, T, E | Biased toward CodeSeek-style layouts | Confirmed |
| 9. Every/all request | Completeness explicitly verified | Normal retrieval/generation | No exhaustive intent or repository-wide enumeration | Grounded but incomplete list | C, T gap | Generic | Confirmed |
| 10. Comparison | Both sides represented and contrasted | Usually semantic retrieval plus narrative LLM | One primary intent; no side-specific subqueries | One-sided comparison | C, fixture limitation | Generic | High confidence |
| 11. Multi-part question | Facet coverage tracked | One primary intent and one candidate pool | No facets or targeted repair retrieval | Parts silently omitted | C, R | Generic | Confirmed |
| 12. Config/version | Manifest is authoritative and retrievable | Config intent and file hints | JSON/YAML excluded from dense/lexical search | Guessed or absent versions | C, R, T | Config-heavy repositories | Confirmed |
| 13. Contradictory sources | Conflict detected and authority applied | Weak file-role/source-truth boosts | No contradiction model | Stale source silently wins | C, test gap | Generic | Confirmed |
| 14. Presence/absence | Absence requires exhaustive evidence | Exact/semantic lookup; special absence route | Top-k failure or zero-scan deterministic route | False repository-wide absence | C | Generic risk plus specific workaround | Confirmed |
| 15. Code extraction | Exact code remains unchanged and cited | Code builder or LLM | Chunk splitting, caps, validation stripping | Truncated or altered code | C, T | Generic within parsed languages | High confidence |
| 16. Behavioral explanation | Definition, callers, config and fallback align | Explanation route plus expansion | Relationship recall and source authority weak | Plausible but incomplete behavior | C, R | Generic | High confidence |
| 17. Broad semantic query | Semantically central evidence dominates | Dense, lexical, semantic boosts | Prefix-biased embeddings and special-case boosts | Relevant but peripheral files | C, E | Generic with repository bias | High confidence |
| 18. Ambiguous query | Clarify or report ambiguity | Confidence floor/`OUT_OF_SCOPE` | Targetless query skips retrieval below 0.50 | Empty/general conversational answer | C, T | Generic | Confirmed |
| 19. Follow-up | Correct prior entities/evidence reused | Entity rewrite and candidate injection | Picks one prior entity; previous files penalized | Wrong antecedent or lost prior source | C, T | Generic | High confidence |
| 20. Topic shift | Previous context excluded | Similarity/overlap topic-shift checks | Vague pronouns force continuation early | Cross-topic contamination | C, T | Generic | High confidence |
| 21. Conversation evidence reuse | Prior facts and paths remain available | One-turn history for follow-ups | History default off; facts tied to displayed sources | Re-retrieval or contradiction with prior answer | C, T | Generic | Confirmed |
| 22. Code/data/config/style span | Each artifact class has a channel | Single merged retrieval | Data/config are excluded or plain-text; caps enforce competition | Cross-layer answer misses one layer | C, R | Mixed-language/apps | Confirmed |
| 23. Numeric/list fidelity | Counts and entries verified | Narrow numeric validation | No list completeness/count contract | Wrong count or missing entries | C, test gap | Generic | Confirmed |
| 24. Insufficient evidence | Partiality or explicit uncertainty | Low-confidence/absence builders | Candidate count substitutes for coverage | Confident false absence or vague answer | C, R | Generic | Confirmed |

---

## E. Ingestion and indexability findings

**Confirmed: the discoverability invariant is not enforced.**

`Every supported authoritative repository artifact must be discoverable through at least one appropriate retrieval channel` is false in the current implementation.

Systemic losses include:

- Lockfiles are filtered, including npm, pnpm, Yarn, Cargo, Poetry, and Gem lockfiles. Exact resolved dependency versions can therefore be unavailable.
- `.github` is excluded, removing authoritative workflow and deployment configuration.
- Real `.env`, databases, PDFs, SVGs, and generated-path classes are excluded. Some exclusions are security- or noise-motivated, but no alternative metadata channel preserves authoritative facts.
- Java, Go, Rust, SQL, XML, Gradle, and several other common artifact types are not supported by the language mapper. They are skipped, not indexed as generic text.
- JSON, YAML, TOML, CSS, Markdown, and shell-like files are mainly file-level text. They lack useful definition/import/export boundaries.
- Only Python and JS/TS families receive AST-like structure. Even there, interfaces, type aliases, constants, exported data structures, dynamic imports, CommonJS imports, and many re-exports are incomplete.
- Every parsed code file gets a full-file chunk as well as symbol chunks. This increases duplicate evidence and lets broad file chunks compete against precise symbol chunks.
- Overflow splitting preserves inherited metadata across parts, but the identifier and defining content can land in different parts.
- Embedding input truncation makes very large source and data files prefix-biased.
- Failed embeddings can be stored as zero vectors. The current live collection had no zero vectors, but there is no production invariant preventing them.
- Incremental state is keyed by path, size, and mtime rather than repository identity, collection identity, content hash, embedding model, or ingestion configuration. A fresh/recreated collection can coexist with “unchanged” files skipped by reused state.
- `repo_sessions.files_indexed` currently reports zero despite 27 relational files, demonstrating that even index-completeness metadata is not reliable.

The live payload sample was internally consistent for its 49 points and contained no zero vectors. That establishes that corruption is not universal; it does not remove the systemic paths above.

---

## F. Query-understanding findings

The processor exposes multiple intent scores but operationally resolves one primary intent. It does not produce requested facets, comparison sides, exhaustiveness requirements, evidence-authority expectations, independent retrieval subqueries, per-facet coverage requirements, required response structure, or negative-claim proof requirements.

Consequences:

- “Compare,” “all,” “every,” “exact,” and multi-part wording mostly influence scattered phrase checks rather than a durable query contract.
- Architecture, dependency, configuration, behavior, and code extraction needs are compressed into a single candidate pool.
- Framework terminology is handled largely through keyword maps and routes.
- Dotted members, aliases, and explicit paths are extracted inconsistently across file types.
- The deterministic classifier is active; the LLM classifier is disabled.
- A targetless query below the `0.50` confidence floor becomes `OUT_OF_SCOPE`, bypassing repository retrieval.

Most severe is the hardcoded absence marker set in `query_processor.py`. Terms related to a narrow trading domain force `ABSENCE_CHECK`, and `main.py` can immediately emit a “comprehensive scan” absence answer without performing that scan. This is both repository-specific bias and a direct violation of negative-claim safety.

---

## G. Retrieval-recall findings

| Channel | Active behavior | Principal limits/failures |
| --- | --- | --- |
| Dense Qdrant | Enabled; 15 candidates | Large-file prefix bias; unsupported artifacts absent; JSON/YAML excluded by active search filters |
| BM25 lexical | Enabled; 15 candidates from cached Qdrant payloads | Same file exclusions; cache limited to 5,000 payloads; invalidation depends on ingestion path |
| Metadata search | File/symbol/path metadata | Metadata completeness depends on parser and payload |
| Exact file search | Qdrant plus local filesystem fallback | Ambiguous basename returns at most two; only indexed/local-bound files |
| Exact symbol search | Payload plus local regex fallback | Fallback limited to Python/JS/JSX/TS/TSX and simple syntax |
| Exact entity search | Extracted file/symbol/service entities | Entity extraction is heuristic and single-target oriented |
| Dependency-edge search | Payload `calls`/imports | Activated narrowly; direction and resolution are incomplete |
| Structural hints | Query phrase/path heuristics | Hint vocabulary is not a general relationship parser |
| Framework routing | FastAPI/React and other keyword routes | Useful as adapters, but some affect global ranking |
| Direct topic injection | CodeSeek-specific topic routes | Familiar-repository optimization; cross-repository bias |
| Previous candidates | Follow-ups only | Maximum 3 and 20%; score ≥0.55; multiplied by 0.85; disabled for several intents |
| Filesystem fallback | Exact local reads | Does not provide general repository-wide semantic/exhaustive search |
| Graph retrieval | Indexed graph exists | Shadow and active modes are both disabled |
| Semantic boost | Labels and keyword overlap | Multiple heuristic boosts are not calibrated as common probabilities |

A current configuration/version trace demonstrates the general mechanism: `.gitignore`, CSS, layout, page, components, and data sources survived, while `package.json` did not. The active search code excludes JSON/YAML/YML/JSONL from both dense and lexical candidates, even though such files may be indexed.

Other recurring recall failures include consumers outranking definitions, definitions without usages, root manifests excluded from search, weak mapping from UI terms to data definitions, semantic matches outranking exact paths, one search formulation for multi-part queries, bounded multi-hop expansion, and weak repository isolation.

Collection names use tenant plus repository basename rather than a full repository identity. Payloads do not independently encode a session/repository identity sufficient to reject contamination. A historical safe-evaluation run expecting CodeSeek files but receiving Portfolio candidates is concrete evidence of this failure class.

---

## H. Structural traversal findings

| Relationship | Ingested/stored | Active retrieval status |
| --- | --- | --- |
| Python absolute/relative imports | Partially extracted/resolved | Import backing/graph capable, graph disabled |
| JS/TS relative imports | Extracted | Import backing active |
| Root `@/` aliases | Special handling | Active in selected paths |
| Arbitrary `tsconfig` aliases | Partial resolver support | Not consistently represented in graph; nested packages weak |
| Barrel exports/re-exports | Limited depth tracing | Not a general graph relationship |
| Named/default exports | Partial | Metadata use varies |
| Dynamic imports | Not reliably extracted | Unavailable |
| CommonJS `require` | Not reliably extracted | Unavailable |
| Index modules | Path heuristics | No robust export graph |
| Class inheritance | Not modeled as graph edges | Unavailable |
| Method-to-parent | Parent metadata | Expansion enabled for recognized methods |
| Function calls | Stored where parser supplies `calls` | Two-hop callee expansion for selected intents |
| Callers | Narrow payload search | Direction/module ambiguity |
| Interface implementations | Not modeled | Unavailable |
| Config-to-runtime | Not modeled | Semantic coincidence/heuristics only |
| Component-to-data | Import backing where supported | Often late and capped |
| Template-to-controller | No generic resolver | Semantic retrieval only |

The graph data is built but not used in the active path. Even if enabled, it currently models only `contains`, `defines`, and `imports`. It cannot answer general caller/callee, inheritance, implementation, runtime-configuration, or template-controller questions.

Search, fusion, filtering, reranking, and top-10 selection occur before optional graph retrieval and metadata expansion. Structural traversal therefore cannot rescue a first-hop file discarded before expansion. Supporting imports are also assigned a late context tier.

---

## I. Fusion and reranking findings

Confirmed behaviors:

- Dense, lexical, metadata, and semantic candidates are fused with RRF using `k=60`.
- Candidate identity is mainly `chunk_id`.
- Lexical scores are normalized separately; dense, synthetic, exact, semantic, and heuristic scores are later combined.
- Exact-like hits can receive boosts around `+10`.
- CodeSeek topic routes can add large fixed boosts.
- Semantic labels use multipliers while other signals use additive boosts or penalties.
- A per-file diversity limit of two applies before final top-10 selection.
- Generic relevance filtering uses `0.45`.
- Pure RRF evidence can remain below that floor unless another boost rescues it.

These scores are not genuinely comparable probabilities. They mix native similarity, BM25, RRF, synthetic exact scores, path/framework heuristics, semantic multipliers, role penalties, and previous-turn penalties.

Recurring consequences include exact lexical coincidences overpowering authority, consumers outranking definitions, configuration receiving insufficient weight, unfamiliar layouts misclassifying documentation as authoritative, tests/templates being suppressed, multiple needed chunks from one file being capped, and support evidence falling below top-10 before expansion.

Stored traces cannot quantify pre-rerank versus post-rerank relevant rank because tracing begins after substantial internal search/fusion/filtering.

---

## J. Authority and contradiction findings

CodeSeek does not have a general source-authority model. Authority is approximated through file/path roles, definition-versus-usage signals, exported-data markers, documentation/test penalties, and prompt instructions.

It does not reliably incorporate freshness, runtime reachability, deprecated status, schema/manifest authority, source agreement, explicit source-of-truth declarations, contradiction detection, or repository-specific build resolution.

CodeSeek-layout rules and Portfolio-oriented source-truth terms are repository-biased. If two sources disagree, one may be removed by ranking or caps, or the model may silently choose. There is no deterministic disagreement record or requirement to surface both sides. No dedicated contradiction or authority-resolution test was found.

---

## K. Source-gating findings

Active lifecycle:

```text
Search/fusion/reranking/top-10
→ optional graph candidates
→ metadata expansion
→ context assembly
→ candidate prioritization
→ display/reasoning split
→ generation
```

`display_sources` is normally capped at 6; `reasoning_sources` is capped at 12.

Observed across 21 stored traces:

- Average retrieved candidates: 8.14.
- Average context candidates: 5.43.
- Average display sources: 2.81.
- Retrieved-to-display survival: 28.51%.
- Only 54.56% of context chunk IDs were user-visible as display sources.
- 90.48% of display-source IDs were represented in recorded context.
- Among non-empty traces, context averaged 6 candidates across 4.11 files.
- Context duplicate chunk-ID ratio: 12.67%.
- Display duplicate chunk-ID ratio: 14.91%.

The model may reason from evidence the user cannot inspect, while a visible citation is not invariably present in recorded model context. Gating optimizes score, role, and diversity rather than comparison sides, facets, definitions plus usages, multi-chunk files, authoritative configuration, or contradictions.

---

## L. Context-assembly findings

Confirmed risks:

- File-level and symbol-level chunks duplicate code.
- The same chunk can appear under different expansion identities.
- Full-file reads can replace or enlarge narrow evidence.
- Primary chunks are truncated before later tiers are considered.
- Supporting imports are late-tier evidence.
- Sources may be recorded when remaining content budget is zero.
- Ordering follows priority/score rather than dependency order.
- Pseudo-XML source wrappers do not escape arbitrary repository content.
- Live filesystem reads can diverge from embedded content.
- History can consume up to 1,500 tokens.

Current traces cannot calculate relevant/irrelevant token ratios, duplicate-token ratios, per-source truncation, requested-facet support, or primary-source truncation frequency.

---

## M. Generation-routing findings

| Query class | Typical route | Systemic routing risk |
| --- | --- | --- |
| Direct file lookup | File summary or exact-source builder | Works only with correctly selected file |
| Direct symbol lookup | Symbol deep-dive/code builder | Same-name ambiguity and missing definition |
| Exact source location | Deterministic location builder | Uses gated visible sources |
| Imported definition | Explanation/LLM | No dedicated import-definition contract |
| Caller/callee | Flow/explanation builder | Builder can outrun relationship evidence |
| Dependency trace | Flow builder or LLM | Partial graph/import chain |
| Multi-hop flow | Deterministic flow or LLM | Bounded expansion presented as complete |
| Architecture overview | Overview/architecture builder | CodeSeek-specific anchors and narratives |
| Exhaustive list | Usually narrative LLM | No exhaustive builder or verifier |
| Comparison | Usually narrative LLM | No two-sided coverage route |
| Multi-part | Route follows one primary intent | Other facets lose |
| Config/version | Config/LLM or low-confidence route | Manifest may never reach route |
| Contradiction | Normal LLM | No contradiction-aware route |
| Presence/absence | Absence or weak-evidence builder | Can declare absence without scan |
| Code extraction | Code builder | Chunk/budget truncation |
| Behavioral explanation | Explanation LLM/builder | Missing cross-file evidence |
| Broad semantic | LLM | Heuristic candidate bias |
| Ambiguous | `OUT_OF_SCOPE` or low context | Retrieval may be skipped rather than clarification |
| Follow-up | Rewritten query then normal route | Wrong entity rewrite |
| Topic shift | Normal route after shift detection | Prior entity contamination |
| Conversation reuse | Follow-up route/history | Facts available only conditionally |
| Code/data/config/style span | One primary route | Cross-layer facets not preserved |
| Numeric/list fidelity | Normal builder/LLM | Narrow numeric validation only |
| Insufficient evidence | Weak-evidence/absence answer | Candidate quantity substitutes for proof |

Sync and streaming share most routing before provider generation, but completion semantics differ. Sync substitutes a non-empty fallback after repeated empty output. Streaming has no equivalent fallback when the stream is syntactically successful but contains no recognized content.

---

## N. Provider and streaming findings

Sync parsing recognizes `message.content`, `reasoning_content`/`reasoning`, `choices[].text`, and some list-based content parts. Streaming recognizes only `choices[0].delta.content`.

Streaming gaps:

- JSON/schema parse errors are silently skipped.
- No stream-event or recognized-text count is enforced.
- No finish-reason classification is retained.
- No final accumulator flush contract exists.
- A zero-content stream does not raise, so sync fallback is not called.
- Deltas reach the browser before post-generation validation.
- The final SSE sequence sends sources and `done`, not a corrected final answer.
- Persistence accepts an empty assistant message.
- The frontend does not process a leftover unterminated buffer after reader close.

Four of twenty current assistant messages are empty. They span at least two query classes, and one had correct hook evidence while another had substantial but incomplete stack evidence. The exact provider/adapter cause remains unresolved because raw outcome classifications and stream-event counts were not persisted.

The invariant “a successful query never returns an empty or whitespace-only answer” is not enforced.

---

## O. Validation and completeness findings

Current validation primarily removes disallowed paths, restricts selected source-location answers, validates fenced code against selected sources, validates numeric literals for limited value-oriented terms, and strips some internal source sections.

It does not validate requested facets, both comparison sides, exhaustive lists, count completeness, contradictions, authority, false absence, partiality, generic factual claims, non-empty generic output, or visible/reasoning citation equivalence.

An answer can therefore be grounded yet severely incomplete. Repair can also omit legitimate reasoning-only filenames, reject numbers from hidden sources, or replace hedged recovery discussion with an unsupported deterministic absence-like statement.

---

## P. False-absence findings

Absence can be inferred from no candidates, low confidence, missing exact targets, LLM interpretation, or deterministic special rules. It is not generally based on repository-wide indexed enumeration, complete metadata, bounded filesystem scanning, completed graph traversal, unsupported-file accounting, or proof of index currency.

The strongest confirmed violation is the hardcoded `ABSENCE_CHECK` route, which can claim a comprehensive scan while returning zero retrieval sources.

Likely false-absence classes include root manifests, filtered lockfiles, unsupported files, large arrays, aliased/CommonJS imports, low-frequency symbols, alternative framework conventions, nested packages, stale incremental state, and same-basename collection collisions.

---

## Q. Confidence-calibration findings

Confidence uses exact-hit status, retrieval score, source/expansion count, and path/symbol overlap. It does not include facet coverage, authority, contradictions, citation alignment, empty output, false-absence risk, list/count completeness, context relevance, or provider outcome.

| Confidence condition | Expected meaning | Current limitation |
| --- | --- | --- |
| Strong/exact | Authoritative evidence fully answers request | Exactness does not prove authority or facet coverage |
| Multiple supporting sources | Independent agreement | Sources may be duplicates or secondary consumers |
| Partial/weak | Evidence is incomplete | No facet model determines incompleteness |
| Empty provider output | Failure | Not included in confidence calculation |
| Absence answer | Exhaustive negative evidence | Can be produced without retrieval |

A trustworthy calibration table cannot be produced from current traces and the contaminated historical evaluation.

---

## R. Conversation-memory findings

Memory is organized around recent displayed source entities, not durable verified facts. Only bounded recent entities are used, history is normally disabled unless follow-up conditions permit it, at most one prior turn is injected, and previous candidates are capped at 3/20%, require score ≥0.55, and receive a `0.85` penalty.

Vague pronouns can force continuation before topic-shift checks. Multiple prior entities can resolve to one incorrect representative. Empty stored answers can participate in later history while cited entities remain available. The visible streamed answer can also differ from the stored post-run answer used by later memory.

---

## S. Generalization and hardcoded-heuristic findings

| Classification | Examples | Assessment |
| --- | --- | --- |
| Generic abstraction | Dense/BM25 fusion, exact paths, bounded context | Broadly reusable but incompletely calibrated |
| Framework adapter | FastAPI and React file/hint rules | Reasonable if isolated and evidence-backed |
| Reasonable heuristic | Prefer definitions, diversify files | Useful but should not replace coverage/authority |
| Repository-specific workaround | Portfolio component/data targeting | Cross-repository scoring bias |
| Self-referential optimization | CodeSeek auth, reports, safe-eval and retrieval-internal routes | Inflates familiar CodeSeek performance |
| Unsafe special rule | Trading-domain absence markers | Can generate false repository-wide absence |
| Layout-specific authority | `backend/retrieval` role assumptions | Misclassifies unfamiliar layouts |

Python and React/TypeScript have the strongest support. CommonJS, nested monorepos, data/config-heavy repositories, nonstandard layouts, and Java/Go/Rust/SQL-heavy systems are substantially weaker.

---

## T. Threshold and budget findings

| Setting | Default/effective | Stage | Purpose | Failure risk | Evidence of tuning |
| ------- | ------: | ----- | ------- | ------------ | ------------------ |
| Dense top-k | 15 | Retrieval | Dense recall | Misses multi-facet/large repos | No broad calibration found |
| Lexical top-k | 15 | Retrieval | Exact-term recall | Same | No broad calibration found |
| Post-merge top-k | 10 | Fusion | Bound work | Evidence removed before expansion | Local tests only |
| RRF constant | 60 | Fusion | Rank fusion | Compresses score differences | Standard heuristic |
| Relevance threshold | 0.45 | Filtering | Remove weak candidates | Pure fused evidence can be removed | No cross-repo tuning evidence |
| Intent floor | 0.50 | Query | Route targetless queries out | Legitimate retrieval skipped | Behavior tests only |
| Follow-up thresholds | 0.72 semantic; 0.15 lexical | Memory | Detect continuation | Missed follow-up or contamination | No calibration report |
| History injection | 0.65 | Memory | Decide reuse | Facts unavailable below threshold | No calibration report |
| Previous candidate | ≥0.55; max 3/20%; ×0.85 | Retrieval | Limit contamination | Correct prior source loses | Tests only |
| Display/reasoning caps | 6 / 12 | Gating | Bound sources | Required evidence removed | Trace statistics only |
| Context cap | 7,000 | Assembly | Model budget | Facet-unaware truncation | No multi-repo study |
| Intent context caps | 1,800–6,500 | Assembly | Route-specific budget | Wrong intent gets wrong budget | No broad tuning |
| History cap/turns | 1,500 / 1 | Memory | Protect code context | Prior facts disappear | No longitudinal eval |
| Graph active | max 2, score ≥90 | Graph | Conservative activation | Disabled; valid graph evidence suppressed | No active calibration |
| Import/flow depth | 3 / 2 | Search/expansion | Bound traversal | Longer chains lost | Synthetic tests |
| Expanded chunks | 6 | Expansion | Context control | Supporting evidence capped | Existing tests |
| Import support | 6 | Search/assembly | Bound definitions | Cross-file evidence lost | Code constant |
| Qdrant timeout | 5 s | Retrieval | Bound latency | Partial/empty recall | Narrow tests |
| Provider timeout | 20 s remote; 120 s local | Generation | Bound latency | Path divergence | Operational defaults |
| Retries/backoff | 3 / 0.5 s×attempt | Retrieval/provider | Recovery | Empty-success bypasses retry | Sync better covered |
| Response cap | 2,048 | Generation | Bound output | Exhaustive answer truncation | No completeness calibration |
| Exact boost | About +10 | Rerank | Protect exact results | Exact irrelevant result dominates | Heuristic tests |
| Expansion decay | 0.8 then 0.5 | Expansion | Prefer primaries | Necessary downstream hop suppressed | Synthetic tests |
| Sibling expansion | Disabled | Expansion | Avoid noise | Adjacent evidence unavailable | Measurement pending |

Most constants have implementation tests but no evidence of empirical tuning across multiple repositories and answer-quality labels.

---

## U. Observability findings

Current traces retain the raw query, some processed intent/rewrite data, candidate lists after internal fusion/filtering, graph-added/reranked/context/final/cited stages, and a partial final-answer preview.

They do not reliably retain complete intent scores/facets, raw per-channel candidates and ranks, fusion/reranker contributions, drop decisions, pre-top-k candidates, expansion failure reasons, per-source context tokens/truncation, pre/post-validation answers, provider outcome classification, stream event counts, finish reasons, retry/fallback sequence, confidence features, or correctness labels.

The first failure point is therefore often inferential. Missing manifests can be tied to active type suppression, but many candidate losses cannot be assigned precisely to retrieval, thresholding, top-k, gating, or truncation. Empty messages are proven, but their exact provider/stream trigger is not.

---

## V. Test and evaluation findings

Safe isolated command run from `backend`:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=. \
CODESEEK_DB_BACKEND=sqlite \
CODESEEK_SQLITE_PATH=/tmp/codeseek-audit-pure.db \
./.venv/bin/python -m pytest -q -p no:cacheprovider --disable-warnings \
  tests/query/test_query_processor_scored_intent.py \
  tests/graph/test_import_resolution.py \
  tests/search/test_python_import_backing.py \
  tests/search/test_two_layer_source_gating.py \
  tests/generation/test_answer_validation.py \
  tests/memory/test_retrieval_followup_resolution.py \
  tests/ingestion/test_filtering.py
```

Result: **106 passed in 4.39 seconds**.

A bounded isolated run of `tests/api/test_query_stream.py` timed out after 25 seconds with code 124. It is inconclusive, not a test failure or pass.

Coverage exists for direct retrieval, selected Python/JS imports, root aliases, limited re-exports, caller/callee helpers, context truncation, confidence mechanics, follow-ups, topic shifts, frameworks, and monorepo fixtures. Explicit quality coverage is missing for contradictions, authority, exhaustive completeness, comparison/facets, false absence, empty provider/stream output, duplicate-context fidelity, and confidence calibration.

The historical safe evaluation is not a valid quality benchmark because its collection contained the wrong repository.

---

## W. Production-invariant matrix

| Required production invariant | Enforced in code? | Enforcement point | Tested? | Known bypass |
| ----------------------------- | ----------------- | ----------------- | ------- | ------------ |
| 1. Successful response is never empty | No | Sync fallback only | No empty-stream test | Successful zero-delta stream |
| 2. Every factual claim is grounded | Partial | Prompt and validators | Partial | Narrative claims and hidden evidence |
| 3. Every facet answered or unavailable | No | None | No | Single primary intent |
| 4. Absence not inferred from top-k failure | No | None | No | Zero-retrieval absence route |
| 5. Contradictory evidence surfaced | No | None | No | Rank/gating silently chooses |
| 6. Implementation outweighs stale secondary material | Partial | Role/source-truth heuristics | Narrow | Unknown layouts |
| 7. Outside-repository content cannot enter context | Partial | Path/session checks | Some | Basename collection collision |
| 8. Displayed citations support answer | Partial | Allowlist/validators | Some | Citation may not be in model context |
| 9. Reasoning evidence has visible citations | No | Two-layer split | Mechanics only | About 45% of context IDs hidden |
| 10. Relevant exact matches survive thresholds | Partial | Exact boosts | Some | Ambiguity and caps |
| 11. Budget does not lose primary evidence | Partial | Token tiers | Truncation tests | Facet-unaware truncation |
| 12. Sync and stream are equivalent | No | Shared upstream route only | Happy path | Parser/fallback divergence |
| 13. Memory does not contaminate shifts | Partial | Similarity/entity checks | Synthetic | Vague pronouns |
| 14. Lists preserve counts/entries | No | Narrow numeric validator | No | No exhaustive contract |
| 15. Provider failure yields fallback/error | Partial | Sync retries; stream exceptions | Partial | Empty successful stream |

---

## X. Systemic root-cause ranking

| Root cause | Pipeline stage | Recurring symptoms | Affected query classes | Affected repository classes | Severity | Evidence strength | Fix complexity |
| ---------- | -------------- | ------------------ | ---------------------- | --------------------------- | -------- | ----------------- | -------------- |
| No facet decomposition or coverage contract | Query → validation | Partial comparisons, lists, multi-part answers | 9–13, 16, 22–24 | All | Critical | Confirmed | High |
| Authoritative artifacts filtered, unsupported, or excluded | Ingestion/retrieval | Missing versions/configs/data | 4, 8, 12–14, 22–24 | Config/data/mixed-language | Critical | Confirmed | Medium |
| Relationships incomplete and graph disabled/late | Ingestion/search/graph | Consumer without definition, shallow flows | 4–7, 16, 22 | Monorepos/cross-file | High | Confirmed | High |
| Rank-based top-k/gating replaces coverage | Fusion/gating | Required evidence disappears | 7–13, 16, 22–24 | Large/multi-layer | Critical | Confirmed | High |
| Streaming lacks non-empty invariant | Provider/API/frontend | Empty persisted responses | All streamed classes | All | Critical | Confirmed gap | Medium |
| Absence asserted without exhaustive evidence | Query/routing/validation | False repository absence | 14, 18, 24 | Nonstandard/config/large | Critical | Confirmed | Medium |
| Incomparable scores and special boosts | Fusion/reranking | Irrelevant candidates dominate | 1–8, 12, 17 | Unfamiliar repos | High | Confirmed | High |
| Weak authority/contradiction heuristics | Reranking/validation | Stale sources treated as truth | 10, 12, 13, 16, 24 | Non-CodeSeek layouts | High | Confirmed | High |
| Weak collection/incremental identity | Ingestion/isolation | Wrong or missing index content | All | Same-basename/multi-session | Critical | High confidence | Medium |
| Trace begins after early loss | Observability | First failure unprovable | All | All | High diagnostic impact | Confirmed | Medium |
| Entity/path-oriented memory | Memory | Poor reuse and contamination | 19–21 | Conversational use | Medium | Confirmed | Medium |
| Hardcoded repository/domain rules | Query/rerank/routing | Familiar-repo inflation and bias | 8, 12, 14, 17 | Unfamiliar repos | High | Confirmed | Medium |

Retrieval plus evidence selection is the dominant correctness and completeness constraint. Provider/stream behavior is a separate critical reliability constraint for empty responses.

---

## Y. Recommended remediation sequence

Recommendations only; no implementation was performed.

| Sequence | Systemic mechanism addressed | Affected query patterns | Expected general benefit | Relevant components | Verification strategy | Risk/trade-off |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | Non-empty stream/result contract | All streamed generation | Eliminates silent empty success | `llm.py`, API, frontend, persistence | Zero-delta, whitespace, malformed, final-buffer and parity tests | Fallback must not disguise outages |
| 2 | Negative-evidence protocol | Presence/absence | Prevents false repository-wide claims | Query, retrieval, absence route, validator | Present-but-low-rank, unsupported, stale-index and alias cases | Repository scans add latency |
| 3 | Explicit facets and coverage | Comparison/exhaustive/multi-part | Enforceable completeness | Query, search, routing, validation | Multi-repo facet-level labels | Risk of over-decomposition |
| 4 | Artifact discoverability invariant | Config/dependency/data | Authoritative files reach a channel | Discovery, languages, parsers, lexical, metadata | Per-artifact contract tests | More noise/storage |
| 5 | Strong repository/index identity | All | Prevents contamination/stale skips | Isolation, payload, state, API | Same-basename and recreated-index tests | Migration/reindex required |
| 6 | Coverage-aware candidate preservation | Multi-hop/multi-part | Preserves evidence roles/facets | Search, fusion, gating, assembler | Per-facet candidate survival | More latency/context |
| 7 | Relationship model and staged retrieval | Imports/calls/flows | Retrieves definitions before truncation | Parser, graph, resolver, expander | Cross-language and real-monorepo fixtures | Graph precision/cost |
| 8 | Calibrated scoring/provenance | Exact/semantic/framework | Reduces heuristic dominance | Searcher, targeting, source truth | Cross-repo ablations | Familiar scores may fall |
| 9 | Authority/contradiction model | Config/behavior/comparison | Surfaces disagreement | Metadata, reranker, validator, prompt | Deliberately contradictory fixtures | Authority is domain-sensitive |
| 10 | Context dedup/logical ordering | Multi-hop/extraction | Better relevant-token ratio | Assembler/filter | Token duplication and coverage metrics | Dedup may remove local context |
| 11 | Fact-oriented memory | Follow-ups/shifts | Safer reuse | Memory, rewrite, history | Multi-turn labeled suites | More state/invalidation |
| 12 | Failure-point observability | All | Empirical diagnosis | Trace/eval tooling | Per-channel ranks/drop reasons/provider classes | Volume/privacy |
| 13 | Isolate special-case rules | Semantic/architecture/absence | Better cross-repo behavior | Query hints, targeting, routing | Cross-repo ablation | Familiar benchmarks may fall |
| 14 | Cross-repo evaluation/calibration | All | Measures real quality | Eval data, confidence, CI | Diverse languages/layouts and facet labels | Maintenance cost |

---

## Z. Unknowns and additional evidence required

The following cannot be proven from current observability:

- Exact cause of the four empty answers.
- Provider event count, finish reason, and recognized/unrecognized deltas.
- Actual pre-validation text and whether it differed from visible streamed text.
- Relevant/irrelevant context-token ratios and truncation frequency.
- Exact first-relevant rank and threshold/top-k/gating losses.
- Confidence calibration against correctness.
- Current behavior on real Java, Go, Rust, SQL, CommonJS-heavy, and large nested-monorepo repositories.
- Production frequency of collection collisions or stale incremental skips.
- Provider timeout/error rates across adapters.
- Evidence that constants were tuned on multiple repositories.
- Active graph accuracy, because graph retrieval is disabled.
- Repository-wide quality metrics, because only one live repository was available and the historical evaluation was contaminated.

The additional evidence needed is clean multi-repository indexes with strong identity, raw per-channel candidate traces, candidate drop reasons, token/truncation accounting, provider outcome classifications, stream-event accounting, pre/post-validation text, facet-level correctness labels, and real cross-language evaluation repositories.

The three most important confirmed systemic root causes affecting general CodeSeek response quality are:

1. **Queries have no explicit facet, completeness, or negative-evidence contract, so retrieval, gating, generation, confidence, and validation cannot know what must be covered.**
2. **Authoritative artifacts and structural relationships are routinely excluded, weakly represented, disabled, or introduced only after irreversible top-k loss.**
3. **The streaming path does not enforce a non-empty final answer and can persist successful empty responses despite usable retrieved evidence.**
