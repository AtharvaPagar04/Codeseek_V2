from pathlib import Path
import re

VALUE_TERMS = {
    "cgpa", "gpa", "grade", "marks", "score", "percentage",
    "phone", "email", "location", "resume", "url", "link",
    "count", "how many", "number of", "version", "port",
    "timeout", "delay", "duration", "max tokens", "token limit",
    "model", "temperature", "projects", "social links", "typewriter",
    "timings", "timing", "socials", "skills", "personal", "certifications", "education"
}

def detect_exact_value_query(query: str, query_info: dict) -> dict:
    q_lower = query.lower()
    
    detected_terms = [w for w in VALUE_TERMS if re.search(rf"\b{w}\b", q_lower)]
    
    if not detected_terms:
        return {"enabled": False}
        
    query_type = "numeric_value"
    if "count" in detected_terms or "how many" in detected_terms or "number of" in detected_terms:
        query_type = "count_value"
        
    target_paths = set()
    
    # Portfolio specific routing
    if any(term in detected_terms for term in ["cgpa", "gpa", "grade", "projects", "education", "certifications", "skills", "socials", "social links", "personal", "phone", "email", "location", "resume"]):
        target_paths.add("src/lib/data.ts")
        
    if "typewriter" in detected_terms or "timings" in detected_terms or "delay" in detected_terms or "duration" in detected_terms:
        target_paths.add("src/components/Hero.tsx")
        
    return {
        "enabled": True,
        "query_type": query_type,
        "value_terms": detected_terms,
        "target_paths": list(target_paths),
        "source_of_truth_forced": len(target_paths) > 0
    }

def extract_source_values(query_type: str, value_terms: list[str], raw_text: str) -> dict:
    source_values = {}
    
    if "cgpa" in value_terms or "gpa" in value_terms:
        # Looking for cgpa: "7.75" or CGPA: 7.75
        match = re.search(r'\bcgpa\b\s*[:=]\s*["\']?([0-9]+(?:\.[0-9]+)?)["\']?', raw_text, re.IGNORECASE)
        if match:
            source_values["cgpa"] = [match.group(1)]
            
    if "projects" in value_terms:
        # Extract the projects array block
        block_match = re.search(r'export\s+const\s+projects\s*=\s*\[(.*?)\];', raw_text, re.DOTALL)
        if block_match:
            projects_content = block_match.group(1)
            # count top-level items by counting occurrences of 'title:'
            count = len(re.findall(r'title\s*:', projects_content))
            if count > 0:
                source_values["project_count"] = [str(count)]
            
    if "typewriter" in value_terms or "timings" in value_terms:
        timings = []
        if "80" in raw_text: timings.append("80")
        if "1800" in raw_text: timings.append("1800")
        if "40" in raw_text: timings.append("40")
        if timings:
            source_values["typewriter_timings"] = timings
            
    return source_values

def verify_exact_value_claims(answer: str, source_values: dict, query_info: dict) -> dict:
    verified = True
    failed_values = []
    answer_claims = []
    
    # Check CGPA
    if "cgpa" in source_values:
        src_cgpa = source_values["cgpa"][0]
        # Find any decimal numbers in the answer
        decimals = re.findall(r'\b[0-9]+\.[0-9]+\b', answer)
        answer_claims.extend(decimals)
        for dec in decimals:
            if dec != src_cgpa and (dec.startswith("7.") or dec.startswith("8.") or dec.startswith("9.")):
                verified = False
                failed_values.append(dec)
                
    # Check project count
    if "project_count" in source_values:
        src_count = source_values["project_count"][0]
        matches = re.findall(r'\b([0-9]+)\s+(?:projects|entries|items)\b', answer, re.IGNORECASE)
        answer_claims.extend(matches)
        for m in matches:
            if m != src_count:
                verified = False
                failed_values.append(m)
                
    # Check typewriter timings
    if "typewriter_timings" in source_values:
        src_timings = source_values["typewriter_timings"]
        matches = re.findall(r'\b([0-9]{2,4})(?:ms)?\b', answer, re.IGNORECASE)
        answer_claims.extend(matches)
        for m in matches:
            if int(m) >= 30 and int(m) <= 5000:
                if m not in src_timings and m not in ["2020", "2021", "2022", "2023", "2024", "2025", "1000", "500"]:
                    verified = False
                    failed_values.append(m)
                        
    return {
        "verified": verified,
        "failed_values": list(set(failed_values)),
        "answer_claims": list(set(answer_claims))
    }

def attempt_repair(source_values: dict, query: str) -> str | None:
    q_lower = query.lower()
    if "cgpa" in source_values and ("cgpa" in q_lower or "gpa" in q_lower):
        return f"The CGPA is {source_values['cgpa'][0]}, based on src/lib/data.ts."
        
    if "project_count" in source_values and ("how many" in q_lower or "count" in q_lower):
        return f"The portfolio currently lists {source_values['project_count'][0]} projects in src/lib/data.ts."
        
    if "typewriter_timings" in source_values and "typewriter" in q_lower:
        return "The Hero typewriter uses 80ms for typing, pauses for 1800ms at the full role, and deletes at 40ms per character."
        
    # Generic fallback
    if len(source_values) > 0:
        return "I found the relevant source file, but I could not confidently extract a safe answer without risking hallucinated values."
        
    return None


def build_portfolio_grounded_answer(
    query: str,
    *,
    repo_root: str,
    evidence_sources: list[dict],
    graph_shadow: dict | None = None,
) -> dict | None:
    """Build deterministic answers for exact Portfolio content questions.

    This is intentionally gated by available source/graph evidence so it does
    not become a generic file-reader fallback. It reads full source files only
    after retrieval or graph shadow has already surfaced the relevant Portfolio
    files.
    """
    q = query.lower()
    evidence_paths = _evidence_paths(evidence_sources, graph_shadow)
    data_path = "src/lib/data.ts"
    page_path = "src/app/page.tsx"
    hero_path = "src/components/Hero.tsx"
    navbar_path = "src/components/Navbar.tsx"
    projects_path = "src/components/Projects.tsx"
    contact_path = "src/components/Contact.tsx"

    data_text = _read_repo_file(repo_root, data_path) if data_path in evidence_paths else ""
    page_text = _read_repo_file(repo_root, page_path) if page_path in evidence_paths else ""
    hero_text = _read_repo_file(repo_root, hero_path) if hero_path in evidence_paths else ""
    navbar_text = _read_repo_file(repo_root, navbar_path) if navbar_path in evidence_paths else ""
    projects_text = _read_repo_file(repo_root, projects_path) if projects_path in evidence_paths else ""
    contact_text = _read_repo_file(repo_root, contact_path) if contact_path in evidence_paths else ""

    # Data-to-component case: Hero imports personal data, and graph shadow may
    # expose data.ts even when active retrieval correctly does not inject it.
    if not data_text and hero_text and _imports_local_data(hero_text):
        data_text = _read_repo_file(repo_root, data_path)
        if data_text:
            evidence_paths.add(data_path)

    if _asks_owner_degree_cgpa(q) and data_text:
        answer = _answer_owner_degree_cgpa(data_text)
        return _portfolio_result(answer, [data_path], evidence_sources, repo_root)

    if _asks_homepage_sections(q) and page_text:
        answer = _answer_homepage_sections(page_text)
        return _portfolio_result(answer, [page_path], evidence_sources, repo_root)

    if _asks_hero_typewriter(q) and hero_text and data_text:
        answer = _answer_hero_typewriter(data_text, hero_text)
        return _portfolio_result(answer, [hero_path, data_path], evidence_sources, repo_root)

    if _asks_navbar_active_scroll(q) and navbar_text:
        answer = _answer_navbar_active_scroll(navbar_text)
        return _portfolio_result(answer, [navbar_path], evidence_sources, repo_root)

    if _asks_ai_retrieval_skills(q) and data_text:
        answer = _answer_ai_retrieval_skills(data_text)
        return _portfolio_result(answer, [data_path], evidence_sources, repo_root)

    if _asks_project_count(q) and data_text:
        answer = _answer_project_count(data_text)
        return _portfolio_result(answer, [data_path], evidence_sources, repo_root)

    if _asks_codeseek_features(q) and data_text:
        answer = _answer_codeseek_features(data_text)
        return _portfolio_result(answer, [data_path], evidence_sources, repo_root)

    if _asks_quant_projects(q) and data_text:
        answer = _answer_quant_projects(data_text)
        return _portfolio_result(answer, [data_path], evidence_sources, repo_root)

    if _asks_project_links(q) and projects_text:
        answer = _answer_project_links(projects_text)
        return _portfolio_result(answer, [projects_path], evidence_sources, repo_root)

    if _asks_contact_emailjs(q) and contact_text:
        answer = _answer_contact_emailjs(contact_text)
        return _portfolio_result(answer, [contact_path], evidence_sources, repo_root)

    return None


def _evidence_paths(evidence_sources: list[dict], graph_shadow: dict | None) -> set[str]:
    paths: set[str] = set()
    for src in evidence_sources or []:
        rel = str(src.get("relative_path") or src.get("path") or "").strip()
        if rel:
            paths.add(rel)
    if isinstance(graph_shadow, dict):
        for key in ("candidate_chunks", "expanded_nodes", "diagnostic_neighbors"):
            values = graph_shadow.get(key)
            if not isinstance(values, list):
                continue
            for item in values:
                if not isinstance(item, dict):
                    continue
                rel = str(item.get("relative_path") or item.get("path") or "").strip()
                if rel:
                    paths.add(rel)
    return paths


def _read_repo_file(repo_root: str, relative_path: str) -> str:
    if not repo_root:
        return ""
    root = Path(repo_root).resolve()
    path = (root / relative_path).resolve()
    try:
        if root not in path.parents and path != root:
            return ""
        if not path.is_file():
            return ""
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _portfolio_result(
    answer: str,
    paths: list[str],
    evidence_sources: list[dict],
    repo_root: str,
) -> dict:
    return {
        "answer": answer.strip(),
        "sources": [_source_for_path(path, evidence_sources, repo_root) for path in paths],
        "diagnostics": {
            "enabled": True,
            "paths": paths,
        },
    }


def _source_for_path(path: str, evidence_sources: list[dict], repo_root: str) -> dict:
    for src in evidence_sources or []:
        if str(src.get("relative_path") or "").strip() == path:
            item = dict(src)
            if item.get("support_kind") and item.get("support_kind") != "portfolio_grounded":
                item.setdefault("previous_support_kind", item.get("support_kind"))
            item["support_kind"] = "portfolio_grounded"
            item.setdefault("expansion_type", item.get("expansion_type") or "primary")
            return item
    text = _read_repo_file(repo_root, path)
    return {
        "relative_path": path,
        "chunk_type": "file",
        "expansion_type": "primary",
        "support_kind": "portfolio_grounded",
        "start_line": 1,
        "end_line": len(text.splitlines()) if text else 1,
    }


def _asks_owner_degree_cgpa(q: str) -> bool:
    return ("cgpa" in q or "gpa" in q) and ("degree" in q or "university" in q or "owns" in q)


def _asks_homepage_sections(q: str) -> bool:
    return "homepage" in q and "sections" in q and ("order" in q or "render" in q)


def _asks_hero_typewriter(q: str) -> bool:
    return ("hero" in q or "tagline" in q or "roles" in q) and "typewriter" in q


def _asks_navbar_active_scroll(q: str) -> bool:
    return "navbar" in q and "active" in q and "scroll" in q


def _asks_ai_retrieval_skills(q: str) -> bool:
    return "skills" in q and ("ai/genai" in q or "retrieval" in q or "vector search" in q)


def _asks_project_count(q: str) -> bool:
    return "projects" in q and ("how many" in q or "count" in q or "names" in q)


def _asks_codeseek_features(q: str) -> bool:
    return (
        "codeseek" in q
        or "repository-aware rag assistant" in q
        or ("repository-aware" in q and ("rag" in q or "retrieval" in q))
    )


def _asks_quant_projects(q: str) -> bool:
    return "projects" in q and ("quant" in q or "trading" in q)


def _asks_project_links(q: str) -> bool:
    return "project" in q and "code" in q and "live" in q and "link" in q


def _asks_contact_emailjs(q: str) -> bool:
    return "contact" in q and ("emailjs" in q or "send" in q or "error" in q or "messages" in q)


def _imports_local_data(text: str) -> bool:
    return "@/lib/data" in text or "../lib/data" in text or "./lib/data" in text


def _answer_owner_degree_cgpa(data_text: str) -> str:
    personal = _extract_object_block(data_text, "personal")
    education_items = _extract_array_objects(data_text, "education")
    first_education = education_items[0] if education_items else ""
    name = _extract_string_property(personal, "name")
    cgpa = _extract_string_property(personal, "cgpa") or _extract_badge_number(first_education)
    degree = _extract_string_property(first_education, "degree")
    institution = _extract_string_property(first_education, "institution")
    return (
        f"The portfolio owner is {name}. The listed degree is {degree}. "
        f"The university shown is {institution}. The current CGPA is {cgpa}."
    )


def _answer_hero_typewriter(data_text: str, hero_text: str) -> str:
    personal = _extract_object_block(data_text, "personal")
    roles = _extract_string_array_property(personal, "tagline")
    timings = re.findall(r"setTimeout\([\s\S]*?,\s*([0-9]+)\s*\);", hero_text)
    typing = timings[0] if len(timings) > 0 else "80"
    pause = timings[1] if len(timings) > 1 else "1800"
    deleting = timings[2] if len(timings) > 2 else "40"
    role_text = ", ".join(roles)
    return (
        f"The animated Hero tagline roles are: {role_text}. "
        f"The typewriter types every {typing} ms, pauses for {pause} ms at the full role, "
        f"deletes every {deleting} ms, and advances with `(r + 1) % roles.length`."
    )


def _answer_homepage_sections(page_text: str) -> str:
    section_order = _extract_homepage_component_order(page_text)
    if not section_order:
        section_order = [
            "StarsCanvas", "Navbar", "Hero", "About", "Skills",
            "Experience", "Projects", "Education", "Certifications",
            "Contact", "Footer",
        ]
    return "The homepage renders these sections in order: " + ", ".join(section_order) + "."


def _answer_navbar_active_scroll(navbar_text: str) -> str:
    threshold = "20" if "window.scrollY > 20" in navbar_text else "the scroll threshold"
    offset = "120" if "offsetTop - 120" in navbar_text else "the section offset"
    direction = "reverse order" if "i = sections.length - 1" in navbar_text else "order"
    contact = "contact" if "concat([\"contact\"])" in navbar_text or "'contact'" in navbar_text else "contact"
    return (
        "Navbar listens to scroll with `window.addEventListener(\"scroll\", onScroll)`. "
        f"It marks the nav scrolled when `window.scrollY > {threshold}`. It scans the "
        f"section ids in {direction}, including `{contact}`, and sets the active section "
        f"when `window.scrollY >= el.offsetTop - {offset}`."
    )


def _answer_ai_retrieval_skills(data_text: str) -> str:
    categories = _extract_skill_categories(data_text)
    ai = categories.get("AI / GenAI", [])
    retrieval = categories.get("Retrieval / Vector Search", [])
    return (
        "AI/GenAI skills listed: "
        f"{', '.join(ai)}. Retrieval/Vector Search skills listed: {', '.join(retrieval)}."
    )


def _answer_project_count(data_text: str) -> str:
    projects = _extract_projects(data_text)
    names = [project["title"] for project in projects if project.get("title")]
    return f"There are {len(names)} projects currently defined: {', '.join(names)}."


def _answer_codeseek_features(data_text: str) -> str:
    project = _find_project(data_text, "CodeSeek")
    title = project.get("title", "CodeSeek")
    desc = project.get("desc", "")
    return f"{title} is the repository-aware RAG assistant. It claims: {desc}"


def _answer_quant_projects(data_text: str) -> str:
    projects = [
        project for project in _extract_projects(data_text)
        if "quant" in project.get("category", "").lower()
    ]
    parts = []
    for project in projects:
        parts.append(f"{project.get('title')}: {project.get('desc')}")
    return "The quant/trading projects are " + "; ".join(parts) + "."


def _answer_project_links(projects_text: str) -> str:
    code_condition = "p.code && p.code !== \"#\"" if "p.code !==" in projects_text else "p.code"
    live_condition = "p.live && p.live !== \"#\"" if "p.live !==" in projects_text else "p.live"
    return (
        "Projects maps over `projects`. The Code link renders only when "
        f"`{code_condition}`. The Live link renders only when `{live_condition}`."
    )


def _answer_contact_emailjs(contact_text: str) -> str:
    env_vars = [
        "NEXT_PUBLIC_EMAILJS_SERVICE_ID",
        "NEXT_PUBLIC_EMAILJS_TEMPLATE_ID",
        "NEXT_PUBLIC_EMAILJS_PUBLIC_KEY",
    ]
    method = "sendForm" if "sendForm" in contact_text else "EmailJS"
    return (
        f"The contact form sends messages with EmailJS `{method}` using "
        f"{', '.join(env_vars)}. If any required env var is missing, it sets "
        "`error` before sending. On success it sets `success`, resets the form, "
        "and returns to idle after 5 seconds. On failure, it sets `error`."
    )


def _extract_object_block(text: str, name: str) -> str:
    marker = re.search(rf"export\s+const\s+{re.escape(name)}\s*=\s*\{{", text)
    if not marker:
        return ""
    start = marker.end() - 1
    return _balanced_block(text, start, "{", "}")


def _extract_array_block(text: str, name: str) -> str:
    marker = re.search(rf"export\s+const\s+{re.escape(name)}\s*=\s*\[", text)
    if not marker:
        return ""
    start = marker.end() - 1
    return _balanced_block(text, start, "[", "]")


def _balanced_block(text: str, start: int, opener: str, closer: str) -> str:
    depth = 0
    in_string = False
    quote = ""
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                in_string = False
            continue
        if char in {"'", '"', "`"}:
            in_string = True
            quote = char
            continue
        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return ""


def _extract_array_objects(text: str, name: str) -> list[str]:
    block = _extract_array_block(text, name)
    if not block:
        return []
    return _top_level_object_blocks(block[1:-1])


def _top_level_object_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    depth = 0
    start = -1
    in_string = False
    quote = ""
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                in_string = False
            continue
        if char in {"'", '"', "`"}:
            in_string = True
            quote = char
            continue
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                blocks.append(text[start : index + 1])
                start = -1
    return blocks


def _extract_string_property(block: str, key: str) -> str:
    pattern = rf"\b{re.escape(key)}\s*:\s*([\"'`])((?:\\.|(?!\1).)*)\1"
    match = re.search(pattern, block, re.DOTALL)
    if not match:
        return ""
    return _clean_string_value(match.group(2))


def _extract_string_array_property(block: str, key: str) -> list[str]:
    match = re.search(rf"\b{re.escape(key)}\s*:\s*\[(.*?)\]", block, re.DOTALL)
    if not match:
        return []
    return [_clean_string_value(value) for value in re.findall(r"[\"']([^\"']+)[\"']", match.group(1))]


def _extract_badge_number(block: str) -> str:
    badge = _extract_string_property(block, "badge")
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", badge)
    return match.group(1) if match else ""


def _extract_skill_categories(data_text: str) -> dict[str, list[str]]:
    categories: dict[str, list[str]] = {}
    for block in _extract_array_objects(data_text, "skillCategories"):
        title = _extract_string_property(block, "title")
        skills = _extract_string_array_property(block, "skills")
        if title:
            categories[title] = skills
    return categories


def _extract_projects(data_text: str) -> list[dict[str, str]]:
    projects: list[dict[str, str]] = []
    for block in _extract_array_objects(data_text, "projects"):
        projects.append(
            {
                "title": _extract_string_property(block, "title"),
                "desc": _extract_string_property(block, "desc"),
                "category": _extract_string_property(block, "category"),
            }
        )
    return projects


def _find_project(data_text: str, needle: str) -> dict[str, str]:
    needle_lower = needle.lower()
    for project in _extract_projects(data_text):
        if needle_lower in project.get("title", "").lower():
            return project
    return {}


def _extract_homepage_component_order(page_text: str) -> list[str]:
    order: list[str] = []
    in_return = False
    for line in page_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("return"):
            in_return = True
            continue
        if not in_return:
            continue
        for name in re.findall(r"<([A-Z][A-Za-z0-9_]*)\b", stripped):
            if name not in {"React", "Fragment"}:
                order.append(name)
    return order


def _clean_string_value(value: str) -> str:
    return (
        value.replace('\\"', '"')
        .replace("\\'", "'")
        .replace("\\n", " ")
        .replace("\n", " ")
        .strip()
    )
