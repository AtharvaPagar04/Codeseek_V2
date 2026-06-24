import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Domain Search Terms Mapping from Phase 3
DOMAIN_SEARCH_TERMS = {
    "domain:auth": [
        "auth", "login", "logout", "session", "cookie", "token",
        "oauth", "credential", "authorization", "authentication", "security"
    ],
    "domain:retrieval": [
        "retrieval", "search", "query", "rerank", "rank", "candidate",
        "context", "source", "filter", "answer", "validation"
    ],
    "domain:ingestion": [
        "ingestion", "index", "indexing", "parse", "parser", "chunk",
        "embed", "embedding", "pipeline", "storage", "repo session"
    ],
    "domain:storage": [
        "storage", "store", "stored", "upsert", "qdrant", "vector", "vectors", "point", "points",
        "payload", "collection", "collections", "delete", "chunk", "chunks", "chunk storage",
        "embedding", "embeddings", "scroll", "client"
    ],
    "domain:configuration": [
        "config", "configuration", "settings", "env", "environment",
        "secret", "key", "variable"
    ],
    "domain:source-filtering": [
        "source", "filter", "filtering", "display source", "selected source",
        "reasoning source", "context pruning", "prune"
    ]
}

# Dynamic Feature Phrase Normalization Map from Phase 5
FEATURE_PHRASE_NORMALIZATION = {
    "source filtering": ["source_filter", "source-filter", "filter_source", "filtering", "source"],
    "exact file context pruning": ["context_pruning", "prune_context", "pruning", "exact_file_pruning"],
    "exact file hits": ["exact_file_hit", "exact_file", "exact_hit"],
    "chunk storage": ["chunk", "chunks", "storage", "store"],
    "qdrant upsert": ["qdrant", "upsert", "vector", "point", "payload"],
    "indexed repo stale": ["stale", "freshness", "dirty", "working tree", "index latest"],
    "working tree freshness": ["freshness", "stale", "dirty", "working_tree", "freshness_status"]
}

class RepoProfile:
    def __init__(self, payloads: list[dict]):
        self.payloads = payloads
        self.files = {}  # relative_path -> metadata dict
        self.symbols = {}  # symbol_name -> list of payloads
        self.paths = []
        
        # Populate files and symbols
        for p in payloads:
            rel_path = p.get("relative_path")
            if not rel_path:
                continue
            
            if rel_path not in self.files:
                self.files[rel_path] = {
                    "relative_path": rel_path,
                    "normalized_path": p.get("normalized_path") or rel_path,
                    "filename": p.get("filename") or rel_path.split("/")[-1],
                    "basename": p.get("basename") or rel_path.split("/")[-1].split(".")[0],
                    "extension": p.get("extension") or (rel_path.split(".")[-1] if "." in rel_path else ""),
                    "language": p.get("language") or "",
                    "defined_symbols": set(),
                    "exported_symbols": set(),
                    "labels": set(),
                    "env_keys": set(),
                    "dependencies": set(),
                    "source_kind": self.classify_source_kind(rel_path, p.get("language"), p.get("labels")),
                    "summaries": [],
                    "code_intents": [],
                }
            
            f_meta = self.files[rel_path]
            
            # Collect symbols
            sym = p.get("symbol_name")
            if sym:
                f_meta["defined_symbols"].add(sym)
                if sym not in self.symbols:
                    self.symbols[sym] = []
                self.symbols[sym].append(p)
                
            q_sym = p.get("qualified_symbol")
            if q_sym:
                f_meta["exported_symbols"].add(q_sym)
                
            # Collect labels
            labels = p.get("labels")
            if labels:
                f_meta["labels"].update(labels)
                
            # Collect env keys/deps
            env_keys = p.get("env_keys")
            if env_keys:
                f_meta["env_keys"].update(env_keys)
            deps = p.get("dependencies")
            if deps:
                f_meta["dependencies"].update(deps)
                
            # Summaries / Intents
            summary = p.get("summary")
            if summary:
                f_meta["summaries"].append(summary)
            intent = p.get("code_intent")
            if intent:
                f_meta["code_intents"].append(intent)
                
        self.paths = list(self.files.keys())
        self._build_framework_profile()

    def _build_framework_profile(self):
        self.framework_profile = {
            "frameworks": [],
            "backend_entrypoints": [],
            "route_registries": [],
            "middleware_files": [],
            "service_files": [],
            "schema_files": [],
            "migration_files": [],
            "test_files": [],
            "frontend_pages": []
        }
        
        has_express = False
        has_react = False
        
        for rel_path, meta in self.files.items():
            path_lower = rel_path.lower()
            filename = meta["filename"].lower()
            
            # Framework detection
            deps = meta.get("dependencies", set())
            if "express" in deps or any("express" in d for d in deps):
                has_express = True
            if "react" in deps or any("react" in d for d in deps):
                has_react = True
                
            if filename in ["app.js", "server.js"] and ("frontend" not in path_lower and "components" not in path_lower):
                has_express = True
            if filename.endswith(".jsx") or filename.endswith(".tsx"):
                has_react = True
                
            # Source roles
            role = "unknown"
            
            if "/test/" in path_lower or "/tests/" in path_lower or filename.endswith(".test.js") or filename.endswith(".spec.js") or filename.endswith(".test.jsx"):
                role = "test"
                self.framework_profile["test_files"].append(rel_path)
            elif "/database/migrations/" in path_lower or "/migrations/" in path_lower:
                role = "migration"
                self.framework_profile["migration_files"].append(rel_path)
            elif ("frontend/src/pages/" in path_lower or "src/pages/" in path_lower or "/pages/" in path_lower) and filename.endswith((".jsx", ".tsx", ".js")):
                role = "frontend_page"
                self.framework_profile["frontend_pages"].append(rel_path)
            elif "/routes/" in path_lower or filename.endswith(".routes.js"):
                role = "route_registry"
                self.framework_profile["route_registries"].append(rel_path)
            elif filename in ["app.js", "server.js", "index.js"] and ("frontend" not in path_lower and "components" not in path_lower and "pages" not in path_lower and "tests" not in path_lower and "docs" not in path_lower):
                role = "backend_entrypoint"
                self.framework_profile["backend_entrypoints"].append(rel_path)
            elif "/middleware/" in path_lower or "middleware" in filename:
                role = "middleware"
                self.framework_profile["middleware_files"].append(rel_path)
            elif "/services/" in path_lower or filename.endswith(".service.js") or "service" in filename:
                role = "service"
                self.framework_profile["service_files"].append(rel_path)
            elif "/controllers/" in path_lower or filename.endswith(".controller.js") or "controller" in filename:
                role = "controller"
            elif "/schemas/" in path_lower or filename.endswith(".schema.js") or "schema" in path_lower:
                role = "schema"
                self.framework_profile["schema_files"].append(rel_path)
            elif "/validators/" in path_lower or filename.endswith(".validator.js"):
                role = "validator"
            elif "/repositories/" in path_lower or filename.endswith(".repository.js"):
                role = "repository"
            elif "/utils/" in path_lower or filename.endswith(".util.js") or "util" in filename:
                role = "utility"
            elif "frontend/src/components" in path_lower or "src/components" in path_lower or "/components/" in path_lower:
                role = "frontend_component"
            elif "docs/" in path_lower or filename.endswith(".md"):
                role = "docs"
            elif "config/" in path_lower or filename.endswith(".env") or "dockerfile" in filename:
                role = "config"
                
            meta["framework_source_role"] = role
            
        if has_express:
            self.framework_profile["frameworks"].append("express")
        if has_react:
            self.framework_profile["frameworks"].append("react")

    def classify_source_kind(self, path: str, language: str | None, labels: list[str] | None) -> str:
        path_lower = path.lower()
        filename = path.split("/")[-1].lower()
        lang = (language or "").lower()
        lbls = [l.lower() for l in (labels or [])]
        
        if "/test/" in path_lower or "/tests/" in path_lower or filename.startswith("test_") or filename.endswith("_test.py"):
            return "tests"
        if "/eval/" in path_lower or "/evals/" in path_lower:
            return "evals"
        if "/docs/" in path_lower or filename.endswith(".md"):
            return "docs"
        if "frontend/src" in path_lower or "src/components" in path_lower or any(ext in lang for ext in ("js", "ts", "tsx", "jsx", "javascript", "typescript")):
            return "frontend"
        if "backend/" in path_lower or lang == "python":
            if "rag_ingestion" in path_lower or "ingestion" in path_lower or any("ingestion" in l for l in lbls):
                return "ingestion"
            if "retrieval" in path_lower or any("retrieval" in l for l in lbls):
                return "retrieval"
            return "backend"
        if "rag_ingestion" in path_lower or "ingestion" in path_lower or any("ingestion" in l for l in lbls):
            return "ingestion"
        if "retrieval" in path_lower or any("retrieval" in l for l in lbls):
            return "retrieval"
        if "artifact:config-file" in lbls or any(pat in filename for pat in ("config", "settings", ".env", "tsconfig")):
            return "config"
            
        return "implementation"

# Cache of profiles by collection name
_profile_cache: dict[str, RepoProfile] = {}

def get_repo_profile(collection: str) -> RepoProfile:
    """Get or build the cached RepoProfile for the given collection name."""
    if collection in _profile_cache:
        return _profile_cache[collection]
    
    # We import searcher helper to scroll payloads
    from retrieval.search.searcher import _scroll_collection_payloads
    logger.info(f"Building repo profile for collection: {collection}")
    payloads = _scroll_collection_payloads(collection)
    profile = RepoProfile(payloads)
    _profile_cache[collection] = profile
    return profile

def compute_dynamic_boosts_and_penalties(item: dict, raw_query: str, entities: dict, collection: str) -> tuple[float, float, dict]:
    if not collection:
        return 0.0, 0.0, {}
        
    try:
        profile = get_repo_profile(collection)
    except Exception:
        return 0.0, 0.0, {}
        
    rel_path = item.get("relative_path", "")
    if not rel_path:
        return 0.0, 0.0, {}
        
    boost = 0.0
    penalty = 0.0
    details = {}
    
    query_lower = raw_query.lower()
    
    # 1. Domain Boost
    boost_labels = entities.get("boost_labels") or []
    domain_terms = set()
    for lbl in boost_labels:
        if lbl in DOMAIN_SEARCH_TERMS:
            domain_terms.update(DOMAIN_SEARCH_TERMS[lbl])
            
    # File metadata from profile
    f_meta = profile.files.get(rel_path)
    if f_meta and domain_terms:
        # Label overlap
        overlap_labels = set(f_meta["labels"]).intersection(boost_labels)
        label_score = len(overlap_labels) * 1.5
        
        # Term overlap in path/filename/basename/defined_symbols
        term_score = 0.0
        path_lower = rel_path.lower()
        filename_lower = f_meta["filename"].lower()
        basename_lower = f_meta["basename"].lower()
        
        for term in domain_terms:
            if term in path_lower:
                term_score += 0.8
            elif term in filename_lower:
                term_score += 0.8
            elif term in basename_lower:
                term_score += 0.8
            for sym in f_meta["defined_symbols"]:
                if term in sym.lower():
                    term_score += 0.5
            for summary in f_meta["summaries"]:
                if term in summary.lower():
                    term_score += 0.1
                    
        content_lower = (item.get("content_excerpt") or item.get("content") or "").lower()
        if content_lower:
            for term in domain_terms:
                if term in content_lower:
                    term_score += 0.1
            for specific_term in ["client.upsert", "pointstruct", "payload", "collection_name"]:
                if specific_term in content_lower:
                    term_score += 0.3

        total_domain_score = label_score + term_score
        if total_domain_score > 0.0:
            boost += min(total_domain_score * 0.12, 0.45)
            
    if item.get("domain_boost_hit"):
        boost += 0.35

    # 2. Feature Phrase Normalization
    matched_features = []
    for phrase, variants in FEATURE_PHRASE_NORMALIZATION.items():
        if phrase in query_lower:
            # Check if candidate matches any variants
            path_lower = rel_path.lower()
            symbol_lower = (item.get("symbol_name") or "").lower()
            summary_lower = (item.get("summary") or "").lower()
            intent_lower = (item.get("code_intent") or "").lower()
            content_lower = (item.get("content_excerpt") or item.get("content") or "").lower()
            
            phrase_match = False
            for var in variants:
                if (
                    var in path_lower
                    or var in symbol_lower
                    or var in summary_lower
                    or var in intent_lower
                    or var in content_lower
                    or var in [l.lower() for l in item.get("labels", [])]
                ):
                    phrase_match = True
                    break
            if phrase_match:
                boost += 0.40
                matched_features.append(phrase)
                
    # 3. Source Kind Boosts and Penalties
    impl_indicators = [
        "where is", "where are", "implemented", "handled", "how does", "how are",
        "explain how", "show me", "explain implementation", "where is done", "work",
        "stored", "done", "protect", "pruning", "targeting", "filtering", "assembled"
    ]
    is_impl_query = any(ind in query_lower for ind in impl_indicators)
    
    # Classify source kind
    kind = profile.classify_source_kind(rel_path, item.get("language"), item.get("labels"))
    
    requests_frontend = any(t in query_lower for t in ["frontend", "ui", "component", "components", "page", "pages", "react", "jsx", "tsx", "display component"])
    requests_tests = any(t in query_lower for t in ["test", "tests", "unit test", "integration test", "eval", "evals", "metrics", "audit", "fixture"])
    requests_docs = any(t in query_lower for t in ["doc", "docs", "documentation", "readme", "plan", "report"])
    
    if is_impl_query:
        is_exact_or_direct = bool(
            item.get("exact_retrieval_hit")
            or item.get("support_kind") in ("direct_injection", "exact_value_forced", "component_definition", "auth_routing", "code_topic_routing")
        )
        if not is_exact_or_direct:
            if kind == "frontend" and not requests_frontend:
                penalty -= 1.80
            elif kind == "tests" and not requests_tests:
                penalty -= 1.80
            elif kind == "evals" and not requests_tests:
                penalty -= 1.80
            elif kind == "docs" and not requests_docs:
                penalty -= 1.00
            elif "/scripts/" in rel_path.lower() and not requests_tests:
                penalty -= 1.00
            elif "fixture" in rel_path.lower() and not requests_tests:
                penalty -= 1.80
            elif "plan" in rel_path.lower() and not requests_docs:
                penalty -= 1.00
            elif "report" in rel_path.lower() and not requests_docs:
                penalty -= 1.00
            
        # Boost preferred kinds
        if kind in {"implementation", "backend", "retrieval", "ingestion", "storage", "config"}:
            boost += 0.25
            
    return boost, penalty, {
        "boost": boost,
        "penalty": penalty,
        "kind": kind,
        "matched_features": matched_features
    }

def build_diagnostics(candidates: list[dict], raw_query: str, entities: dict, collection: str) -> dict:
    boost_labels = entities.get("boost_labels") or []
    
    # Extract active domain terms
    domain_terms = set()
    for lbl in boost_labels:
        if lbl in DOMAIN_SEARCH_TERMS:
            domain_terms.update(DOMAIN_SEARCH_TERMS[lbl])
            
    candidate_paths = []
    boosted_paths = []
    penalized_paths = []
    source_kind_penalties = []
    
    # Determine which source kinds were penalized
    query_lower = raw_query.lower()
    impl_indicators = [
        "where is", "implemented", "handled", "how does", "how are",
        "show me", "explain implementation", "where is done", "work",
        "stored", "done"
    ]
    is_impl_query = any(ind in query_lower for ind in impl_indicators)
    requests_frontend = any(t in query_lower for t in ["frontend", "ui", "component", "components", "page", "pages", "react", "jsx", "tsx"])
    requests_tests = any(t in query_lower for t in ["test", "tests", "unit test", "integration test", "eval", "evals", "metrics", "audit"])
    requests_docs = any(t in query_lower for t in ["doc", "docs", "documentation", "readme"])
    
    if is_impl_query:
        if not requests_frontend:
            source_kind_penalties.append("frontend")
        if not requests_tests:
            source_kind_penalties.extend(["tests", "evals"])
        if not requests_docs:
            source_kind_penalties.append("docs")
            
    for item in candidates:
        rel_path = item.get("relative_path")
        if not rel_path:
            continue
            
        candidate_paths.append(rel_path)
        
        # Calculate what boost/penalty this candidate got
        boost, penalty, details = compute_dynamic_boosts_and_penalties(item, raw_query, entities, collection)
        if boost > 0.0:
            boosted_paths.append(rel_path)
        if penalty < 0.0:
            penalized_paths.append(rel_path)
            
    # Check if exact hits are preserved
    exact_hits_preserved = any(item.get("exact_retrieval_hit") for item in candidates)
    
    return {
        "enabled": len(boost_labels) > 0,
        "boost_labels": list(boost_labels),
        "domain_terms": list(domain_terms),
        "candidate_paths": sorted(list(set(candidate_paths)))[:10],
        "boosted_paths": sorted(list(set(boosted_paths)))[:10],
        "penalized_paths": sorted(list(set(penalized_paths)))[:10],
        "source_kind_penalties": source_kind_penalties,
        "exact_hits_preserved": exact_hits_preserved
    }

def generate_feature_recall_terms(query: str) -> list[str]:
    stopwords = {"where", "how", "what", "is", "are", "does", "do", "from", "being", "for", "in", "the", "an", "a", "of", "to", "and", "on", "at", "by", "with", "about"}
    intentwords = {"done", "implemented", "handled", "assembled", "located", "defined", "work", "protected", "validate", "handle", "dropped", "responses", "audited", "show", "me", "explain", "code", "file"}
    words = re.findall(r"[a-z0-9_]+", query.lower())
    terms = [w for w in words if w not in stopwords and w not in intentwords and len(w) >= 3]
    
    variants = set()
    for t in terms:
        variants.add(t)
        if t.endswith("ing"): variants.add(t[:-3])
        elif t.endswith("ed"): variants.add(t[:-2])
        elif t.endswith("s") and not t.endswith("ss"): variants.add(t[:-1])
        
    for i in range(len(terms) - 1):
        w1, w2 = terms[i], terms[i+1]
        variants.add(f"{w1}_{w2}")
        variants.add(f"{w2}_{w1}")
        variants.add(f"{w1}-{w2}")
        
        s1 = w1[:-3] if w1.endswith("ing") else (w1[:-2] if w1.endswith("ed") else (w1[:-1] if w1.endswith("s") and not w1.endswith("ss") else w1))
        s2 = w2[:-3] if w2.endswith("ing") else (w2[:-2] if w2.endswith("ed") else (w2[:-1] if w2.endswith("s") and not w2.endswith("ss") else w2))
        
        if s1 != w1 or s2 != w2:
            variants.add(f"{s1}_{s2}")
            variants.add(f"{s2}_{s1}")
            variants.add(f"{s1}-{s2}")
            
    return sorted(list(variants))

def discover_feature_recall_candidates(
    query: str,
    repo_profile: RepoProfile,
    limit: int = 5,
) -> list[dict]:
    query_lower = query.lower()
    impl_indicators = [
        "where is", "where are", "implemented", "handled", "how does", "how are",
        "explain how", "show me", "explain implementation", "where is done", "work",
        "stored", "done", "protect", "pruning", "targeting", "filtering", "assembled", "audited", "validate"
    ]
    is_impl_query = any(ind in query_lower for ind in impl_indicators)
    if not is_impl_query:
        return []
        
    terms = generate_feature_recall_terms(query)
    if not terms:
        return []
        
    candidates = []
    
    for rel_path, f_meta in repo_profile.files.items():
        score = 0.0
        matched_terms = []
        
        path_lower = rel_path.lower()
        basename_lower = f_meta["basename"].lower()
        
        for term in terms:
            term_score = 0.0
            if term == basename_lower:
                term_score += 4.0
            elif term in basename_lower:
                term_score += 2.0
            elif term in path_lower:
                term_score += 1.0
                
            for sym in f_meta["defined_symbols"]:
                if term in sym.lower():
                    term_score += 1.5
                    break
            
            for summary in f_meta["summaries"]:
                if term in summary.lower():
                    term_score += 0.5
                    break
                    
            for intent in f_meta["code_intents"]:
                if term in intent.lower():
                    term_score += 0.5
                    break
                    
            for lbl in f_meta["labels"]:
                if term in lbl.lower():
                    term_score += 0.5
                    break
                    
            if term_score > 0:
                score += term_score
                matched_terms.append(term)
                
        if score > 0:
            kind = f_meta["source_kind"]
            if kind in ["backend", "retrieval", "ingestion", "implementation"]:
                score += 1.0
            elif kind in ["frontend", "tests", "evals", "docs"]:
                score -= 2.0
                
            unique_terms = set(matched_terms)
            if len(unique_terms) > 1:
                score += 2.0
                
            if score >= 2.0:
                candidates.append({
                    "relative_path": rel_path,
                    "score": score,
                    "matched_terms": list(unique_terms)
                })
                
    candidates.sort(key=lambda x: x["score"], reverse=True)
    
    results = []
    for c in candidates[:limit]:
        # Find first payload matching this path
        payload = next((p for p in repo_profile.payloads if p.get("relative_path") == c["relative_path"]), None)
        if payload:
            p_copy = dict(payload)
            p_copy["feature_recall_hit"] = True
            p_copy["support_kind"] = "feature_recall"
            p_copy["feature_recall_terms"] = c["matched_terms"]
            p_copy["feature_recall_score"] = c["score"]
            results.append(p_copy)
            
    return results
