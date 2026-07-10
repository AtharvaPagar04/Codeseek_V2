"""Constants, registry, and configuration for chunk labeling."""

from __future__ import annotations

# Confidence thresholds
STRONG_MATCH = 0.90
MEDIUM_MATCH = 0.70
WEAK_MATCH = 0.55

MIN_CONFIDENCE = 0.50
MAX_CONFIDENCE = 0.95

# Max labels permitted per category
MAX_LABELS_PER_CATEGORY = {
    "artifact": 3,
    "code_role": 1,
    "domain": 3,
    "capability": 3,
    "tech": 3,
    "question_use": 4,
}

MAX_TOTAL_LABELS = 12

# All valid labels and their descriptions
LABEL_REGISTRY = {
    # artifact
    "artifact:source-code": "Source code files containing functions, methods, or classes.",
    "artifact:repo-summary": "High-level summary of the entire repository.",
    "artifact:readme": "Project README or documentation overview.",
    "artifact:documentation": "General documentation or markdown files.",
    "artifact:product-doc": "Product-level documentation under docs/product/.",
    "artifact:package-manifest": "Project package manifest (e.g. package.json, requirements.txt).",
    "artifact:dockerfile": "Docker build configuration file.",
    "artifact:docker-compose": "Docker compose multi-container orchestration configuration.",
    "artifact:env-example": "Template for environment variable configuration.",
    "artifact:test-code": "Test suites, integration tests, or unit tests.",

    # code_role
    "code_role:function": "Standalone function definition.",
    "code_role:method": "Class method definition.",
    "code_role:class": "Class structure definition.",

    # domain
    "domain:auth": "Authentication and authorization logic (tokens, sessions, OAuth).",
    "domain:retrieval": "Information retrieval and ranking logic (CodeSeek internal).",
    "domain:ingestion": "Repository parsing, chunking, and indexing pipeline (CodeSeek internal).",
    "domain:provider-management": "LLM provider and API key management (CodeSeek internal).",
    "domain:frontend": "User interface pages, components, or styles.",
    "domain:testing": "Validation, unit tests, or test execution framework.",
    "domain:devops": "CI/CD, containerization, or deployment setup.",
    "domain:vector-db": "Vector database interface and queries.",
    "domain:documentation": "Documentation, guides, and technical writing.",
    "domain:product": "Product-level documentation, roadmaps, and feature overviews.",

    # capability
    "capability:dependency-management": "Third-party libraries and package tracking.",
    "capability:qdrant-storage": "Reading or writing vector records in Qdrant.",
    "capability:vector-upsert": "Preparing or inserting points into the vector database.",
    "capability:embedding-generation": "Generating vector embeddings for search.",
    "capability:live-indexing-events": "Real-time updates or server-sent events for indexing.",
    "capability:session-validation": "Verifying active user sessions.",
    "capability:token-validation": "Validating authentication tokens.",

    # tech
    "tech:docker": "Docker containerization and orchestration.",
    "tech:qdrant": "Qdrant vector search engine.",
    "tech:sentence-transformers": "SentenceTransformers embedding library.",
    "tech:sse": "Server-Sent Events for streaming.",

    # question_use
    "question_use:technical-explanation": "Explaining how specific technical logic works.",
    "question_use:code-location": "Locating files or symbols in the project.",
    "question_use:code-snippet": "Showing actual code blocks as examples.",
    "question_use:implementation": "Target for editing, modifying, or refactoring code.",
    "question_use:repo-overview": "General understanding of project scope and architecture.",
    "question_use:general-context": "Background knowledge for general queries.",
    "question_use:architecture": "Explaining system design, module roles, and component relationships.",
    "question_use:setup": "Installing, configuring, or running the application.",
    "question_use:dependency-question": "Questions about packages, versions, or libraries.",
    "question_use:config-question": "Questions about settings, environments, or parameters.",
    "question_use:test-validation": "Validating code correctness or test outcomes.",
    "question_use:debugging": "Troubleshooting errors or unexpected behaviors.",
}

# CodeSeek internal-only labels
CODESEEK_INTERNAL_LABELS = {
    "domain:retrieval",
    "domain:ingestion",
    "domain:provider-management",
}

# Config flags
from rag_ingestion.config import ENABLE_CHUNK_LABELS
