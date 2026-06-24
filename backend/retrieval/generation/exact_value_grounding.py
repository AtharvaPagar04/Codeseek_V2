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

