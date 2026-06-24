"""Summary generation stage."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

from rag_ingestion.models.chunk import Chunk


def generate_summary(chunk: Chunk) -> str:
    """Generate a deterministic AST-based chunk summary."""
    if chunk.chunk_type == "function":
        lines = [f"Function: {chunk.symbol_name}"]
        if chunk.parameters:
            lines.append(f"Parameters: {', '.join(chunk.parameters)}")
        if chunk.docstring:
            lines.append(f"Docstring: {chunk.docstring}")
        return "\n".join(lines)

    if chunk.chunk_type == "method":
        lines = [f"Method: {chunk.symbol_name}", f"Class: {chunk.parent_symbol}"]
        if chunk.parameters:
            lines.append(f"Parameters: {', '.join(chunk.parameters)}")
        if chunk.docstring:
            lines.append(f"Docstring: {chunk.docstring}")
        return "\n".join(lines)

    if chunk.chunk_type == "class":
        lines = [f"Class: {chunk.symbol_name}"]
        if chunk.methods:
            lines.append(f"Methods: {', '.join(chunk.methods)}")
        if chunk.docstring:
            lines.append(f"Docstring: {chunk.docstring}")
        return "\n".join(lines)

    if chunk.chunk_type == "file":
        lines = [f"File: {chunk.relative_path}"]
        extra = _structured_file_summary(chunk)
        if extra:
            lines.append(extra)
        if chunk.file_symbols:
            lines.append(f"Symbols: {', '.join(chunk.file_symbols)}")
        return "\n".join(lines)

    return ""


def _structured_file_summary(chunk: Chunk) -> str:
    relative_path = chunk.relative_path.lower().replace("\\", "/")
    filename = relative_path.split("/")[-1]
    content = chunk.content

    if filename.startswith("readme"):
        _extract_readme_metadata(chunk, content)
    elif filename == "package.json":
        _extract_package_json_metadata(chunk, content)
    elif filename == "requirements.txt":
        _extract_requirements_metadata(chunk, content)
    elif filename == "pyproject.toml":
        _extract_pyproject_metadata(chunk, content)
    elif filename in ("docker-compose.yml", "docker-compose.yaml"):
        _extract_docker_compose_metadata(chunk, content)
    elif filename == "dockerfile":
        _extract_dockerfile_metadata(chunk, content)
    elif (filename.endswith(".env.example") or filename == ".env.example" or
          (filename.startswith(".env") and (filename.endswith(".example") or "example" in filename))):
        _extract_env_example_metadata(chunk, content)
    elif filename == "tsconfig.json":
        _extract_tsconfig_metadata(chunk, content)
    elif filename in ("next.config.js", "next.config.ts", "next.config.mjs"):
        _extract_next_config_metadata(chunk, content)
    elif filename in ("vite.config.js", "vite.config.ts", "vite.config.mjs"):
        _extract_vite_config_metadata(chunk, content)
    elif filename in ("tailwind.config.js", "tailwind.config.ts", "tailwind.config.mjs"):
        _extract_tailwind_config_metadata(chunk, content)
    elif filename in ("postcss.config.js", "postcss.config.mjs"):
        _extract_postcss_config_metadata(chunk, content)
    elif filename in ("eslint.config.js", "eslint.config.mjs"):
        _extract_eslint_config_metadata(chunk, content)
    elif filename == "vercel.json":
        _extract_vercel_metadata(chunk, content)
    elif filename == "netlify.toml":
        _extract_netlify_metadata(chunk, content)
    elif filename == "turbo.json":
        _extract_turbo_metadata(chunk, content)
    elif filename == "caddyfile":
        _extract_caddyfile_metadata(chunk, content)
    elif filename == "nginx.conf":
        _extract_nginx_metadata(chunk, content)
    elif filename in ("config.py", "settings.py") or filename.endswith((".ini", ".lock")):
        _extract_generic_config_metadata(chunk, content)
    elif filename in ("jsconfig.json", "pnpm-workspace.yaml", "render.yaml", "railway.json") or \
            filename.endswith((".json", ".yaml", ".yml", ".toml", ".conf", ".mjs", ".cjs")):
        _extract_generic_config_metadata(chunk, content)
        
    return " | ".join(chunk.summary_facts[:8])


def _is_noise_readme_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    
    # 1. Badge or image markdown, e.g. [![...](...)](...) or ![...](...)
    if re.match(r"^!?\[[^\]]*\]\([^)]+\)", stripped):
        return True
    if stripped.startswith("!["):
        return True
        
    # 2. Raw URLs or only links (e.g. [Link Text](http...))
    if re.match(r"^https?://[^\s]+$", stripped):
        return True
    if re.match(r"^\[[^\]]+\]\([^\)]+\)$", stripped):
        return True
        
    # 3. Copyright lines, e.g., Copyright (c), ©
    if "copyright" in stripped.lower() or "©" in stripped:
        return True
        
    # 4. License lines, e.g., Licensed under, MIT License
    if "license" in stripped.lower():
        return True
        
    return False


def _extract_readme_metadata(chunk, content: str) -> None:
    chunk.file_type = "readme"
    headings = _readme_sections(content)
    
    purpose = None
    lines = content.splitlines()
    
    # Try Priority 2: Description under headings like Overview/About/Introduction
    for heading in ("overview", "about", "introduction"):
        for h, section_lines in headings.items():
            if heading in h.lower():
                for line in section_lines:
                    if not _is_noise_readme_line(line) and len(line.split()) >= 5:
                        purpose = line.strip().rstrip(".")
                        break
            if purpose:
                break
        if purpose:
            break
            
    # Try Priority 1: First non-empty paragraph after H1 title
    if not purpose:
        h1_found = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("# ") or (stripped.startswith("#") and not stripped.startswith("##")):
                h1_found = True
                continue
            if h1_found:
                if stripped and not stripped.startswith("#"):
                    if not _is_noise_readme_line(stripped) and len(stripped.split()) >= 5:
                        purpose = stripped.rstrip(".")
                        break
                elif stripped.startswith("#"):
                    # Hit another heading without finding a paragraph
                    break

    # Try Priority 3: Existing fallback
    if not purpose:
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#") or not stripped:
                continue
            if not _is_noise_readme_line(stripped) and len(stripped.split()) >= 5:
                purpose = stripped.rstrip(".")
                break

    # Try Priority 4: Fallback to H1 heading title
    if not purpose:
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("# ") or (stripped.startswith("#") and not stripped.startswith("##")):
                purpose = stripped.lstrip("#").strip()
                break
                
    if purpose:
        chunk.purpose = purpose
        chunk.summary_facts.append(f"Overview: {chunk.purpose}")
        
    chunk.setup_steps = _section_commands(headings, ("install", "setup", "getting started"))
    chunk.usage_commands = _section_commands(headings, ("usage", "run", "development"))
    chunk.architecture_notes = _section_lines(headings, ("architecture", "structure", "design"), limit=4)
    if chunk.setup_steps:
        chunk.summary_facts.append(f"Setup commands: {', '.join(chunk.setup_steps[:4])}")
    if chunk.usage_commands:
        chunk.summary_facts.append(f"Usage commands: {', '.join(chunk.usage_commands[:4])}")
    if chunk.architecture_notes:
        chunk.summary_facts.append(f"Architecture notes: {'; '.join(chunk.architecture_notes[:2])}")


def _extract_package_json_metadata(chunk: Chunk, content: str) -> None:
    chunk.file_type = "package_json"
    try:
        payload = json.loads(content)
    except Exception:
        return
    name = str(payload.get("name", "")).strip()
    description = str(payload.get("description", "")).strip()
    chunk.dependencies = _dedupe(list((payload.get("dependencies") or {}).keys()))
    chunk.dev_dependencies = _dedupe(list((payload.get("devDependencies") or {}).keys()))
    chunk.scripts = {str(k): str(v) for k, v in (payload.get("scripts") or {}).items()}
    chunk.detected_frameworks = _detect_frameworks(chunk.dependencies + chunk.dev_dependencies)
    chunk.config_tools = _detect_config_tools(chunk.dependencies + chunk.dev_dependencies + list(chunk.scripts))
    
    # Package manager extraction
    pm = None
    pm_field = payload.get("packageManager")
    if pm_field and isinstance(pm_field, str):
        pm = pm_field.split("@")[0].strip()
    if not pm:
        script_cmds = " ".join(chunk.scripts.values()).lower()
        if "pnpm" in script_cmds:
            pm = "pnpm"
        elif "yarn" in script_cmds:
            pm = "yarn"
        elif "bun" in script_cmds:
            pm = "bun"
        elif "npm" in script_cmds:
            pm = "npm"
    if not pm and chunk.file_path:
        parent = Path(chunk.file_path).parent
        if (parent / "pnpm-lock.yaml").exists():
            pm = "pnpm"
        elif (parent / "yarn.lock").exists():
            pm = "yarn"
        elif (parent / "package-lock.json").exists():
            pm = "npm"
        elif (parent / "bun.lockb").exists() or (parent / "bun.lock").exists():
            pm = "bun"
            
    if pm:
        chunk.package_manager = pm
        chunk.summary_facts.append(f"Package manager: {pm}")

    # Role hints
    deps_and_dev_deps = chunk.dependencies + chunk.dev_dependencies
    frameworks_detected = []
    if "next" in deps_and_dev_deps:
        frameworks_detected.append("Next.js app")
    if "react" in deps_and_dev_deps:
        frameworks_detected.append("React app")
    if "vue" in deps_and_dev_deps or "nuxt" in deps_and_dev_deps:
        frameworks_detected.append("Vue app")
    if "svelte" in deps_and_dev_deps or "sveltekit" in deps_and_dev_deps:
        frameworks_detected.append("Svelte app")
    if "angular" in deps_and_dev_deps:
        frameworks_detected.append("Angular app")
        
    if "nestjs" in deps_and_dev_deps or "@nestjs/core" in deps_and_dev_deps:
        frameworks_detected.append("NestJS backend")
    if "express" in deps_and_dev_deps:
        frameworks_detected.append("Express backend")
    if "fastify" in deps_and_dev_deps:
        frameworks_detected.append("Fastify backend")
    if "hono" in deps_and_dev_deps:
        frameworks_detected.append("Hono backend")
        
    if frameworks_detected:
        chunk.summary_facts.append(f"Runtime: {', '.join(frameworks_detected)}")

    # Entrypoints
    for field in ("main", "module", "exports", "bin"):
        val = payload.get(field)
        if val:
            if isinstance(val, dict):
                for k, v in val.items():
                    if isinstance(v, str):
                        chunk.entrypoints.append(v)
            elif isinstance(val, str):
                chunk.entrypoints.append(val)
    chunk.entrypoints = _dedupe(chunk.entrypoints)

    # Tooling
    tooling = []
    for tool in ("eslint", "prettier", "typescript", "tailwindcss", "postcss", "vite", "webpack"):
        if any(tool in dep for dep in deps_and_dev_deps):
            tooling.append(tool)
    if tooling:
        chunk.summary_facts.append(f"Tooling: {', '.join(tooling)}")

    if name:
        chunk.summary_facts.append(f"Package: {name}")
    if description:
        chunk.summary_facts.append(f"Description: {description.rstrip('.')}")
    if chunk.dependencies or chunk.dev_dependencies:
        deps = _dedupe(chunk.dependencies[:4] + chunk.dev_dependencies[:4])
        chunk.summary_facts.append(f"Dependencies: {', '.join(deps[:8])}")
    if chunk.scripts:
        chunk.summary_facts.append(f"Scripts: {', '.join(list(chunk.scripts)[:8])}")
    if chunk.detected_frameworks:
        chunk.summary_facts.append(f"Frameworks: {', '.join(chunk.detected_frameworks[:8])}")


def _extract_requirements_metadata(chunk: Chunk, content: str) -> None:
    chunk.file_type = "requirements"
    packages = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        packages.append(re.split(r"(==|>=|<=|~=)", stripped, maxsplit=1)[0].strip())
    chunk.dependencies = _dedupe(packages)
    chunk.detected_frameworks = _detect_frameworks(chunk.dependencies)
    if chunk.dependencies:
        chunk.summary_facts.append(f"Python dependencies: {', '.join(chunk.dependencies[:8])}")
    if chunk.detected_frameworks:
        chunk.summary_facts.append(f"Frameworks: {', '.join(chunk.detected_frameworks[:8])}")


def _extract_pyproject_metadata(chunk: Chunk, content: str) -> None:
    chunk.file_type = "pyproject"
    try:
        payload = tomllib.loads(content)
    except Exception:
        return
    project = payload.get("project") or {}
    build_system = payload.get("build-system") or {}
    name = str(project.get("name", "")).strip()
    deps: list[str] = []
    for item in project.get("dependencies") or []:
        deps.append(_normalize_dependency_name(str(item)))
    optional = project.get("optional-dependencies") or {}
    dev_deps: list[str] = []
    for values in optional.values():
        for item in values or []:
            dev_deps.append(_normalize_dependency_name(str(item)))
    poetry = ((payload.get("tool") or {}).get("poetry") or {})
    if poetry:
        deps.extend(str(key) for key in (poetry.get("dependencies") or {}) if key.lower() != "python")
        dev_deps.extend(str(key) for key in ((poetry.get("group") or {}).get("dev") or {}).get("dependencies") or {})
    chunk.dependencies = _dedupe(deps)
    chunk.dev_dependencies = _dedupe(dev_deps)
    chunk.build_system = ", ".join(str(item) for item in build_system.get("requires") or [])
    chunk.config_tools = _detect_pyproject_tools(payload)
    chunk.detected_frameworks = _detect_frameworks(chunk.dependencies + chunk.dev_dependencies)
    if name:
        chunk.summary_facts.append(f"Project: {name}")
    if chunk.dependencies:
        chunk.summary_facts.append(f"Dependencies: {', '.join(chunk.dependencies[:8])}")
    if chunk.dev_dependencies:
        chunk.summary_facts.append(f"Dev dependencies: {', '.join(chunk.dev_dependencies[:8])}")
    if chunk.build_system:
        chunk.summary_facts.append(f"Build system: {chunk.build_system}")
    if chunk.config_tools:
        chunk.summary_facts.append(f"Config tools: {', '.join(chunk.config_tools[:8])}")


def _compose_single_value(lines: list[str], key: str) -> str | None:
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(f"{key}:"):
            val = stripped.split(":", 1)[1].strip().strip("'\"")
            if val:
                return val
    return None


def _extract_docker_compose_metadata(chunk: Chunk, content: str) -> None:
    chunk.file_type = "docker_compose"
    service_blocks = _compose_service_blocks(content)
    chunk.services = list(service_blocks)
    service_dependencies: dict[str, list[str]] = {}
    images = []
    builds = []
    
    for service, lines in service_blocks.items():
        service_dependencies[service] = _compose_list_values(lines, "depends_on")
        chunk.ports.extend(_compose_list_values(lines, "ports"))
        chunk.volumes.extend(_compose_list_values(lines, "volumes"))
        chunk.env_keys.extend(_compose_env_keys(lines))
        
        img = _compose_single_value(lines, "image")
        if img:
            images.append(f"{service} ({img})")
        bld = _compose_single_value(lines, "build")
        if bld:
            builds.append(f"{service} ({bld})")
            
    chunk.service_dependencies = {
        key: value for key, value in service_dependencies.items() if value
    }
    chunk.ports = _dedupe(chunk.ports)
    chunk.volumes = _dedupe(chunk.volumes)
    chunk.env_keys = _dedupe(chunk.env_keys)
    if chunk.services:
        chunk.summary_facts.append(f"Services: {', '.join(chunk.services[:8])}")
    if chunk.ports:
        chunk.summary_facts.append(f"Ports: {', '.join(chunk.ports[:8])}")
    if chunk.env_keys:
        chunk.summary_facts.append(f"Environment keys: {', '.join(chunk.env_keys[:8])}")
    if chunk.volumes:
        chunk.summary_facts.append(f"Volumes: {', '.join(chunk.volumes[:8])}")
    if images:
        chunk.summary_facts.append(f"Images: {', '.join(images[:6])}")
    if builds:
        chunk.summary_facts.append(f"Builds: {', '.join(builds[:6])}")


def _extract_dockerfile_metadata(chunk: Chunk, content: str) -> None:
    chunk.file_type = "dockerfile"
    copied_files = []
    
    for line in content.splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("FROM ") and not chunk.base_image:
            chunk.base_image = stripped[5:].strip()
        elif upper.startswith("WORKDIR "):
            chunk.workdir = stripped[8:].strip()
        elif upper.startswith("EXPOSE "):
            chunk.ports.extend(stripped[7:].split())
        elif upper.startswith(("CMD ", "ENTRYPOINT ")):
            chunk.entrypoints.append(stripped)
        elif upper.startswith("COPY ") or upper.startswith("ADD "):
            parts = stripped.split()
            if len(parts) > 1:
                copied_files.append(parts[1])
                
        if "pnpm install" in stripped:
            chunk.package_manager = "pnpm"
        elif "npm install" in stripped or "npm ci" in stripped:
            chunk.package_manager = "npm"
        elif "yarn install" in stripped:
            chunk.package_manager = "yarn"
        elif "pip install" in stripped or "pip3 install" in stripped:
            chunk.package_manager = "pip"
            
    runtimes = []
    base_lower = (chunk.base_image or "").lower()
    if "node" in base_lower or chunk.package_manager in ("npm", "pnpm", "yarn"):
        runtimes.append("node")
    if "python" in base_lower or chunk.package_manager in ("pip",):
        runtimes.append("python")
    if "nginx" in base_lower:
        runtimes.append("nginx")
        
    if chunk.base_image:
        chunk.summary_facts.append(f"Base image: {chunk.base_image}")
    if chunk.workdir:
        chunk.summary_facts.append(f"Workdir: {chunk.workdir}")
    if chunk.ports:
        chunk.summary_facts.append(f"Ports: {', '.join(_dedupe(chunk.ports)[:8])}")
    if chunk.entrypoints:
        chunk.summary_facts.append(f"Entrypoints: {', '.join(chunk.entrypoints[:4])}")
    if chunk.package_manager:
        chunk.summary_facts.append(f"Package manager: {chunk.package_manager}")
    if copied_files:
        chunk.summary_facts.append(f"Copied files: {', '.join(_dedupe(copied_files)[:6])}")
    if runtimes:
        chunk.summary_facts.append(f"Runtime hints: {', '.join(_dedupe(runtimes))}")


def _extract_env_example_metadata(chunk: Chunk, content: str) -> None:
    chunk.file_type = "env_example"
    keys = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key = stripped.split("=", 1)[0].strip()
        keys.append(key)
    chunk.env_keys = _dedupe(keys)
    chunk.feature_flags = [key for key in chunk.env_keys if key.endswith(("_ENABLED", "_ENABLE")) or "ENABLE" in key or "FEATURE" in key]
    chunk.provider_keys = [key for key in chunk.env_keys if any(term in key for term in ("API_KEY", "TOKEN", "SECRET", "PASSWORD"))]
    if chunk.env_keys:
        chunk.summary_facts.append(f"Environment keys: {', '.join(chunk.env_keys[:8])}")
    if chunk.feature_flags:
        chunk.summary_facts.append(f"Feature flags: {', '.join(chunk.feature_flags[:8])}")
    if chunk.provider_keys:
        chunk.summary_facts.append(f"Provider/secret keys: {', '.join(chunk.provider_keys[:8])}")


def _clean_json_comments(text: str) -> str:
    # remove single-line comments // ...
    text = re.sub(r'//.*', '', text)
    # remove block comments /* ... */
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    return text


def _extract_tsconfig_metadata(chunk: Chunk, content: str) -> None:
    chunk.file_type = "tsconfig"
    chunk.config_tools = _dedupe(chunk.config_tools + ["typescript"])
    chunk.summary_facts.append("Tooling: typescript")
    try:
        clean = _clean_json_comments(content)
        payload = json.loads(clean)
    except Exception:
        payload = {}
    
    opts = payload.get("compilerOptions", {})
    target = opts.get("target")
    module = opts.get("module")
    
    if target:
        chunk.summary_facts.append(f"Compiler target: {target}")
    if module:
        chunk.summary_facts.append(f"Compiler module: {module}")


def _extract_next_config_metadata(chunk: Chunk, content: str) -> None:
    chunk.file_type = "next_config"
    chunk.detected_frameworks = _dedupe(chunk.detected_frameworks + ["Next.js"])
    chunk.config_tools = _dedupe(chunk.config_tools + ["next"])
    chunk.summary_facts.append("Framework: Next.js")
    
    if re.search(r"reactStrictMode\s*:\s*true", content):
        chunk.summary_facts.append("React strict mode: enabled")
    if re.search(r"output\s*:\s*['\"]export['\"]", content):
        chunk.summary_facts.append("Output: static export")


def _extract_vite_config_metadata(chunk: Chunk, content: str) -> None:
    chunk.file_type = "vite_config"
    chunk.config_tools = _dedupe(chunk.config_tools + ["vite"])
    chunk.summary_facts.append("Tooling: vite")
    
    plugins = []
    if "react(" in content or "@vitejs/plugin-react" in content:
        chunk.detected_frameworks = _dedupe(chunk.detected_frameworks + ["React"])
        plugins.append("react")
    if "vue(" in content or "@vitejs/plugin-vue" in content:
        chunk.detected_frameworks = _dedupe(chunk.detected_frameworks + ["Vue"])
        plugins.append("vue")
    if "svelte(" in content or "@sveltejs/vite-plugin-svelte" in content:
        chunk.detected_frameworks = _dedupe(chunk.detected_frameworks + ["Svelte"])
        plugins.append("svelte")
        
    if plugins:
        chunk.summary_facts.append(f"Vite plugins: {', '.join(plugins)}")


def _extract_tailwind_config_metadata(chunk: Chunk, content: str) -> None:
    chunk.file_type = "tailwind_config"
    chunk.detected_frameworks = _dedupe(chunk.detected_frameworks + ["Tailwind CSS"])
    chunk.config_tools = _dedupe(chunk.config_tools + ["tailwindcss"])
    chunk.summary_facts.append("Tooling: tailwindcss")
    
    content_matches = re.findall(r"['\"](\./[^'\"]+)['\"]", content)
    if content_matches:
        paths = _dedupe([p for p in content_matches if "*" in p or "." in p])
        if paths:
            chunk.summary_facts.append(f"Tailwind content paths: {', '.join(paths[:3])}")


def _extract_postcss_config_metadata(chunk: Chunk, content: str) -> None:
    chunk.file_type = "postcss_config"
    chunk.config_tools = _dedupe(chunk.config_tools + ["postcss"])
    chunk.summary_facts.append("Tooling: postcss")
    
    plugins = []
    if "tailwindcss" in content:
        chunk.detected_frameworks = _dedupe(chunk.detected_frameworks + ["Tailwind CSS"])
        plugins.append("tailwindcss")
    if "autoprefixer" in content:
        plugins.append("autoprefixer")
    if plugins:
        chunk.summary_facts.append(f"PostCSS plugins: {', '.join(plugins)}")


def _extract_eslint_config_metadata(chunk: Chunk, content: str) -> None:
    chunk.file_type = "eslint_config"
    chunk.config_tools = _dedupe(chunk.config_tools + ["eslint"])
    chunk.summary_facts.append("Tooling: eslint")
    
    if "typescript-eslint" in content or "@typescript-eslint" in content:
        chunk.config_tools = _dedupe(chunk.config_tools + ["typescript"])
        chunk.summary_facts.append("ESLint config: TypeScript supported")


def _extract_vercel_metadata(chunk: Chunk, content: str) -> None:
    chunk.file_type = "vercel_config"
    chunk.config_tools = _dedupe(chunk.config_tools + ["vercel"])
    chunk.summary_facts.append("Deployment: vercel")
    try:
        payload = json.loads(content)
    except Exception:
        payload = {}
    
    if "framework" in payload:
        chunk.summary_facts.append(f"Vercel framework: {payload['framework']}")
    if "routes" in payload:
        chunk.summary_facts.append(f"Vercel routes: {len(payload['routes'])} rules")
    elif "rewrites" in payload:
        chunk.summary_facts.append(f"Vercel rewrites: {len(payload['rewrites'])} rules")


def _extract_netlify_metadata(chunk: Chunk, content: str) -> None:
    chunk.file_type = "netlify_config"
    chunk.config_tools = _dedupe(chunk.config_tools + ["netlify"])
    chunk.summary_facts.append("Deployment: netlify")
    try:
        payload = tomllib.loads(content)
    except Exception:
        payload = {}
        
    build = payload.get("build", {})
    cmd = build.get("command")
    publish = build.get("publish")
    
    if cmd:
        chunk.summary_facts.append(f"Netlify build command: {cmd}")
    if publish:
        chunk.summary_facts.append(f"Netlify publish directory: {publish}")


def _extract_turbo_metadata(chunk: Chunk, content: str) -> None:
    chunk.file_type = "turbo_config"
    chunk.config_tools = _dedupe(chunk.config_tools + ["turbo"])
    chunk.summary_facts.append("Tooling: turborepo")
    try:
        clean = _clean_json_comments(content)
        payload = json.loads(clean)
    except Exception:
        payload = {}
        
    tasks = list(payload.get("tasks", payload.get("pipeline", {})))
    if tasks:
        chunk.summary_facts.append(f"Turbo tasks: {', '.join(tasks[:6])}")


def _extract_caddyfile_metadata(chunk: Chunk, content: str) -> None:
    chunk.file_type = "caddyfile"
    chunk.config_tools = _dedupe(chunk.config_tools + ["caddy"])
    chunk.summary_facts.append("Web server: caddy")
    
    hosts = []
    proxies = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "{" in stripped and not stripped.startswith(("reverse_proxy", "root", "file_server")):
            parts = stripped.split("{")[0].strip().split()
            if parts:
                hosts.extend(parts)
        if "reverse_proxy" in stripped:
            parts = stripped.split("reverse_proxy")[1].strip().split()
            if parts:
                proxies.append(parts[0])
                
    if hosts:
        chunk.summary_facts.append(f"Caddy hosts: {', '.join(hosts[:4])}")
    if proxies:
        chunk.summary_facts.append(f"Caddy proxies: {', '.join(proxies[:4])}")


def _extract_nginx_metadata(chunk: Chunk, content: str) -> None:
    chunk.file_type = "nginx_config"
    chunk.config_tools = _dedupe(chunk.config_tools + ["nginx"])
    chunk.summary_facts.append("Web server: nginx")
    
    ports = []
    server_names = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("listen ") and stripped.endswith(";"):
            port = stripped.split("listen ")[1].replace(";", "").strip().split()[0]
            ports.append(port)
        if stripped.startswith("server_name ") and stripped.endswith(";"):
            names = stripped.split("server_name ")[1].replace(";", "").strip().split()
            server_names.extend(names)
            
    if ports:
        chunk.summary_facts.append(f"Nginx listen ports: {', '.join(_dedupe(ports)[:4])}")
    if server_names:
        chunk.summary_facts.append(f"Nginx server names: {', '.join(_dedupe(server_names)[:4])}")


def _extract_generic_config_metadata(chunk: Chunk, content: str) -> None:
    relative_path = chunk.relative_path.lower()
    
    if "jsconfig.json" in relative_path:
        chunk.file_type = "jsconfig"
        chunk.config_tools = _dedupe(chunk.config_tools + ["javascript"])
        chunk.summary_facts.append("Tooling: jsconfig")
    elif "pnpm-workspace.yaml" in relative_path or "pnpm-workspace.yml" in relative_path:
        chunk.file_type = "pnpm_workspace"
        chunk.config_tools = _dedupe(chunk.config_tools + ["pnpm"])
        chunk.summary_facts.append("Workspace: pnpm")
    elif "render.yaml" in relative_path or "render.yml" in relative_path:
        chunk.file_type = "render_config"
        chunk.summary_facts.append("Deployment: render")
    elif "railway.json" in relative_path:
        chunk.file_type = "railway_config"
        chunk.summary_facts.append("Deployment: railway")
    else:
        chunk.file_type = "config"
        
        tools = []
        if ".json" in relative_path:
            tools.append("json")
        elif ".yaml" in relative_path or ".yml" in relative_path:
            tools.append("yaml")
        elif ".toml" in relative_path:
            tools.append("toml")
        elif ".conf" in relative_path:
            tools.append("conf")
        elif ".mjs" in relative_path or ".cjs" in relative_path or ".js" in relative_path or ".ts" in relative_path:
            tools.append("javascript")
            
        chunk.config_tools = _dedupe(chunk.config_tools + tools)
        chunk.summary_facts.append(f"Config file: {chunk.relative_path}")


def _readme_sections(content: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current = "root"
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            current = stripped.lstrip("#").strip().lower()
            sections.setdefault(current, [])
            continue
        if stripped:
            sections.setdefault(current, []).append(stripped)
    return sections


def _section_lines(sections: dict[str, list[str]], names: tuple[str, ...], limit: int = 8) -> list[str]:
    lines: list[str] = []
    for heading, values in sections.items():
        if any(name in heading for name in names):
            for value in values:
                cleaned = value.strip("-*` ")
                if cleaned and not cleaned.startswith("```"):
                    lines.append(cleaned)
                if len(lines) >= limit:
                    return lines
    return lines


def _section_commands(sections: dict[str, list[str]], names: tuple[str, ...]) -> list[str]:
    commands = []
    for line in _section_lines(sections, names, limit=16):
        cleaned = line.strip("` ")
        if re.search(r"\b(npm|pnpm|yarn|pip|uv|python|docker|docker compose|pytest|uvicorn)\b", cleaned):
            commands.append(cleaned)
    return _dedupe(commands)


def _detect_frameworks(names: list[str]) -> list[str]:
    mapping = {
        "fastapi": "FastAPI",
        "flask": "Flask",
        "django": "Django",
        "litestar": "Litestar",
        "express": "Express",
        "fastify": "Fastify",
        "nestjs": "NestJS",
        "hono": "Hono",
        "koa": "Koa",
        
        "react": "React",
        "next": "Next.js",
        "next.js": "Next.js",
        "vite": "Vite",
        "vue": "Vue",
        "nuxt": "Nuxt",
        "svelte": "Svelte",
        "angular": "Angular",
        "astro": "Astro",
        "remix": "Remix",
        
        "prisma": "Prisma",
        "drizzle-orm": "Drizzle ORM",
        "mongoose": "Mongoose",
        "sequelize": "Sequelize",
        "typeorm": "TypeORM",
        "sqlalchemy": "SQLAlchemy",
        "alembic": "Alembic",
        "psycopg": "Psycopg",
        "asyncpg": "Asyncpg",
        "redis": "Redis",
        "ioredis": "ioredis",
        
        "qdrant-client": "Qdrant",
        "sentence-transformers": "Sentence Transformers",
        "langchain": "LangChain",
        "llama-index": "LlamaIndex",
        "openai": "OpenAI",
        "anthropic": "Anthropic",
        "google-generativeai": "Google Generative AI",
        "groq": "Groq",
        
        "tailwindcss": "Tailwind CSS",
        "postcss": "PostCSS",
        "eslint": "ESLint",
        "prettier": "Prettier",
        "typescript": "TypeScript",
        "webpack": "Webpack",
        "turborepo": "Turborepo",
    }
    
    detected = []
    for name in names:
        lowered = str(name).lower()
        for key, value in mapping.items():
            if key in lowered:
                detected.append(value)
        if "@nestjs/" in lowered:
            detected.append("NestJS")
        if "drizzle" in lowered:
            detected.append("Drizzle ORM")
        if "qdrant" in lowered:
            detected.append("Qdrant")
            
    return _dedupe(detected)


def _detect_config_tools(names: list[str]) -> list[str]:
    possible_tools = {
        "eslint": "eslint",
        "prettier": "prettier",
        "typescript": "typescript",
        "tailwindcss": "tailwindcss",
        "postcss": "postcss",
        "vite": "vite",
        "webpack": "webpack",
        "next": "next",
        "turbo": "turbo",
        "jest": "jest",
        "vitest": "vitest",
        "pytest": "pytest",
        "ruff": "ruff",
        "mypy": "mypy",
        "black": "black",
        "uv": "uv",
        "docker": "docker",
        "caddy": "caddy",
        "nginx": "nginx",
        "vercel": "vercel",
        "netlify": "netlify"
    }
    
    tools = []
    for name in names:
        lowered = str(name).lower()
        for key, value in possible_tools.items():
            if key in lowered:
                tools.append(value)
    return _dedupe(tools)


def _detect_pyproject_tools(payload: dict) -> list[str]:
    tool = payload.get("tool") or {}
    tools = list(tool)
    build_system = payload.get("build-system") or {}
    for item in build_system.get("requires") or []:
        tools.append(_normalize_dependency_name(str(item)))
    return _dedupe(tools)


def _normalize_dependency_name(value: str) -> str:
    return (
        value.split("[", 1)[0]
        .split("==", 1)[0]
        .split(">=", 1)[0]
        .split("<=", 1)[0]
        .split("~=", 1)[0]
        .strip()
    )


def _compose_service_blocks(content: str) -> dict[str, list[str]]:
    blocks: dict[str, list[str]] = {}
    current = ""
    in_services = False
    for line in content.splitlines():
        if not in_services:
            if line.strip() == "services:":
                in_services = True
            continue
        if not line.strip():
            continue
        if not line.startswith("  "):
            break
        if line.startswith("  ") and not line.startswith("    ") and ":" in line:
            current = line.strip().rstrip(":")
            blocks[current] = []
            continue
        if current:
            blocks[current].append(line)
    return blocks


def _compose_list_values(lines: list[str], key: str) -> list[str]:
    values: list[str] = []
    in_key = False
    for line in lines:
        stripped = line.strip().strip("'\"")
        if stripped.startswith(f"{key}:"):
            remainder = stripped.split(":", 1)[1].strip()
            if remainder and remainder != "[]":
                values.extend(_inline_list_values(remainder))
            in_key = True
            continue
        if in_key:
            if not line.startswith("      "):
                in_key = False
                continue
            item = stripped.lstrip("-").strip().strip("'\"")
            if item:
                values.append(item)
    return _dedupe(values)


def _inline_list_values(value: str) -> list[str]:
    stripped = value.strip().strip("[]")
    if not stripped:
        return []
    return [part.strip().strip("'\"") for part in stripped.split(",") if part.strip()]


def _compose_env_keys(lines: list[str]) -> list[str]:
    keys: list[str] = []
    in_env = False
    for line in lines:
        stripped = line.strip().strip("'\"")
        if stripped.startswith("environment:"):
            remainder = stripped.split(":", 1)[1].strip()
            if remainder and remainder.startswith("["):
                keys.extend(item.split("=", 1)[0] for item in _inline_list_values(remainder))
            in_env = True
            continue
        if in_env:
            if not line.startswith("      "):
                in_env = False
                continue
            item = stripped.lstrip("-").strip().strip("'\"")
            if "=" in item:
                keys.append(item.split("=", 1)[0].strip())
            elif ":" in item:
                keys.append(item.split(":", 1)[0].strip())
            elif item:
                keys.append(item)
    return _dedupe(keys)


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        cleaned = str(value).strip()
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        result.append(cleaned)
    return result
