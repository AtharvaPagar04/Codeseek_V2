import re

BEHAVIOR_WORDS = {
    "logic", "typewriter", "render", "rendered", "rendering", "map", "cards", "tags", "grid", "state", "effect",
    "animation", "active", "section", "scroll", "cta", "social", "buttons", "how", "explain", "does", "works", "implemented"
}

MOUNTING_WORDS = {
    "mounted", "used", "where", "homepage", "composition", "structure", "layout", "route"
}

KNOWN_COMPONENTS = {
    "hero", "projects", "navbar", "starscanvas", "skills", "experience", "education", "certifications", "contact",
    "footer", "about", "backtotop"
}

def detect_component_semantic_targets(query: str, query_info: dict, candidates: list[dict] | None = None) -> dict:
    candidates = candidates or []
    q_lower = query.lower()
    
    # 1. Extract tokens that could be components
    tokens = re.findall(r"[A-Za-z_]+", query)
    component_candidates = set()
    
    # PascalCase or known components
    for t in tokens:
        if t.lower() in KNOWN_COMPONENTS:
            # Reconstruct PascalCase for known components if they were typed in lowercase
            pascal = t[0].upper() + t[1:].lower()
            if t.lower() == "starscanvas":
                pascal = "StarsCanvas"
            elif t.lower() == "backtotop":
                pascal = "BackToTop"
            component_candidates.add(pascal)
        elif len(t) > 2 and t[0].isupper() and any(c.islower() for c in t):
            component_candidates.add(t)
            
    if not component_candidates:
        return {"enabled": False}
        
    has_behavior = any(re.search(rf"\b{w}\b", q_lower) for w in BEHAVIOR_WORDS)
    has_mounting = any(re.search(rf"\b{w}\b", q_lower) for w in MOUNTING_WORDS)
    
    # We only enable strict semantic targeting if it asks about behavior OR is clearly asking about the component
    if not has_behavior and has_mounting:
        return {
            "enabled": False,
            "symbols_detected": list(component_candidates),
            "target_paths": [],
            "target_reason": "mounting_query",
            "confidence": "low",
            "ambiguous": False
        }
        
    # Build candidate paths from candidates list
    indexed_paths = {c.get("relative_path", "") for c in candidates if c.get("relative_path")}
    indexed_symbols = {(c.get("symbol_name", ""), c.get("relative_path", "")) for c in candidates if c.get("symbol_name")}
    
    target_paths = set()
    reason = ""
    ambiguous = False
    
    for comp in component_candidates:
        # 1. Metadata symbol definition match
        symbol_matches = [p for s, p in indexed_symbols if s == comp]
        if symbol_matches:
            target_paths.add(symbol_matches[0])
            reason = "metadata_symbol_match"
            continue
            
        # 2. Indexed filename basename match
        basename_matches = [p for p in indexed_paths if p.split("/")[-1].startswith(comp + ".")]
        if len(basename_matches) == 1:
            target_paths.add(basename_matches[0])
            reason = "indexed_filename_match"
            continue
        elif len(basename_matches) > 1:
            # Try to find exactly comp.tsx or comp.ts
            exact = [p for p in basename_matches if p.split("/")[-1] in (f"{comp}.tsx", f"{comp}.ts", f"{comp}.jsx", f"{comp}.js")]
            if len(exact) == 1:
                target_paths.add(exact[0])
                reason = "exact_filename_match"
                continue
            ambiguous = True
            continue
            
        # 3. Guess common paths
        common = [
            f"src/components/{comp}.tsx",
            f"src/components/{comp}.ts",
            f"src/components/{comp}/index.tsx",
            f"src/components/{comp}/index.ts",
            f"components/{comp}.tsx",
            f"components/{comp}.ts"
        ]
        guessed = [p for p in common if p in indexed_paths]
        if guessed:
            target_paths.add(guessed[0])
            reason = "guessed_path_match"
            continue
            
    if not target_paths or ambiguous:
        return {
            "enabled": False,
            "symbols_detected": list(component_candidates),
            "target_paths": list(target_paths),
            "target_reason": "ambiguous" if ambiguous else "no_paths_resolved",
            "confidence": "low",
            "ambiguous": ambiguous
        }
        
    return {
        "enabled": True,
        "symbols_detected": list(component_candidates),
        "target_paths": list(target_paths),
        "target_reason": reason,
        "confidence": "high",
        "ambiguous": False
    }
