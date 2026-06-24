import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from rag_ingestion.models.chunk import Chunk
from rag_ingestion.stages.summary import (
    _structured_file_summary,
    _detect_frameworks,
)


class SummaryExtractionTests(unittest.TestCase):
    def test_readme_purpose_ignores_noise(self) -> None:
        content = (
            "# CodeSeek\n"
            "[![Build Status](https://img.shields.io/travis/user/repo.svg)](https://travis-ci.org/user/repo)\n"
            "![Logo](/logo.png)\n"
            "https://my-live-url.com\n"
            "\n"
            "CodeSeek is a powerful semantic search tool.\n"
            "\n"
            "Copyright (c) 2026 Acme Corp. Licensed under MIT License.\n"
        )
        chunk = Chunk(
            relative_path="README.md",
            chunk_type="file",
            content=content,
        )
        _structured_file_summary(chunk)
        self.assertEqual(chunk.purpose, "CodeSeek is a powerful semantic search tool")
        self.assertIn("Overview: CodeSeek is a powerful semantic search tool", chunk.summary_facts)

    def test_package_json_detections(self) -> None:
        content = """{
  "name": "my-app",
  "packageManager": "pnpm@9.0.0",
  "scripts": {
    "dev": "next dev",
    "build": "next build"
  },
  "dependencies": {
    "next": "^14.0.0",
    "react": "^18.2.0"
  },
  "devDependencies": {
    "typescript": "^5.0.0",
    "tailwindcss": "^3.3.0"
  }
}"""
        
        with TemporaryDirectory() as tmp:
            file_path = Path(tmp) / "package.json"
            file_path.write_text(content, encoding="utf-8")
            # Create a fake lockfile to verify path resolution
            (Path(tmp) / "pnpm-lock.yaml").write_text("", encoding="utf-8")
            
            chunk = Chunk(
                file_path=str(file_path),
                relative_path="package.json",
                chunk_type="file",
                content=content,
            )
            _structured_file_summary(chunk)
            
            self.assertEqual(chunk.file_type, "package_json")
            self.assertEqual(chunk.package_manager, "pnpm")
            self.assertIn("Next.js", chunk.detected_frameworks)
            self.assertIn("React", chunk.detected_frameworks)
            self.assertIn("tailwindcss", chunk.config_tools)
            self.assertIn("typescript", chunk.config_tools)
            self.assertEqual(chunk.scripts, {"dev": "next dev", "build": "next build"})
            
            facts_str = " ".join(chunk.summary_facts)
            self.assertIn("Package manager: pnpm", facts_str)
            self.assertIn("Runtime: Next.js app, React app", facts_str)
            self.assertIn("Tooling: typescript, tailwindcss", facts_str)

    def test_tsconfig_json_extraction(self) -> None:
        content = """{
  // compiler settings
  "compilerOptions": {
    "target": "es2022",
    "module": "esnext"
  }
}"""
        chunk = Chunk(
            relative_path="tsconfig.json",
            chunk_type="file",
            content=content,
        )
        _structured_file_summary(chunk)
        self.assertEqual(chunk.file_type, "tsconfig")
        self.assertIn("typescript", chunk.config_tools)
        self.assertIn("Tooling: typescript", chunk.summary_facts)
        self.assertIn("Compiler target: es2022", chunk.summary_facts)
        self.assertIn("Compiler module: esnext", chunk.summary_facts)

    def test_next_config_extraction(self) -> None:
        content = """
module.exports = {
  reactStrictMode: true,
  output: 'export'
};
"""
        chunk = Chunk(
            relative_path="next.config.mjs",
            chunk_type="file",
            content=content,
        )
        _structured_file_summary(chunk)
        self.assertEqual(chunk.file_type, "next_config")
        self.assertIn("Next.js", chunk.detected_frameworks)
        self.assertIn("React strict mode: enabled", chunk.summary_facts)
        self.assertIn("Output: static export", chunk.summary_facts)

    def test_eslint_config_extraction(self) -> None:
        content = """
import ts from 'typescript-eslint';
export default [
  ...ts.configs.recommended
];
"""
        chunk = Chunk(
            relative_path="eslint.config.mjs",
            chunk_type="file",
            content=content,
        )
        _structured_file_summary(chunk)
        self.assertEqual(chunk.file_type, "eslint_config")
        self.assertIn("eslint", chunk.config_tools)
        self.assertIn("typescript", chunk.config_tools)
        self.assertIn("ESLint config: TypeScript supported", chunk.summary_facts)

    def test_postcss_config_extraction(self) -> None:
        content = """
module.exports = {
  plugins: {
    tailwindcss: {},
    autoprefixer: {}
  }
};
"""
        chunk = Chunk(
            relative_path="postcss.config.mjs",
            chunk_type="file",
            content=content,
        )
        _structured_file_summary(chunk)
        self.assertEqual(chunk.file_type, "postcss_config")
        self.assertIn("postcss", chunk.config_tools)
        self.assertIn("Tailwind CSS", chunk.detected_frameworks)
        self.assertIn("PostCSS plugins: tailwindcss, autoprefixer", chunk.summary_facts)

    def test_tailwind_config_extraction(self) -> None:
        content = """
module.exports = {
  content: ["./src/**/*.{js,ts,jsx,tsx}"],
  theme: { extend: {} }
};
"""
        chunk = Chunk(
            relative_path="tailwind.config.ts",
            chunk_type="file",
            content=content,
        )
        _structured_file_summary(chunk)
        self.assertEqual(chunk.file_type, "tailwind_config")
        self.assertIn("Tailwind CSS", chunk.detected_frameworks)
        self.assertIn("tailwindcss", chunk.config_tools)
        self.assertIn("Tailwind content paths: ./src/**/*.{js,ts,jsx,tsx}", chunk.summary_facts)

    def test_docker_compose_extraction(self) -> None:
        content = """version: '3.8'
services:
  web:
    build: .
    ports:
      - "80:80"
    environment:
      - PORT=80
      - DB_URL=postgresql://localhost
    depends_on:
      - db
  db:
    image: postgres:15
    volumes:
      - pgdata:/var/lib/postgresql/data"""
        chunk = Chunk(
            relative_path="docker-compose.yml",
            chunk_type="file",
            content=content,
        )
        _structured_file_summary(chunk)
        self.assertEqual(chunk.file_type, "docker_compose")
        self.assertEqual(chunk.services, ["web", "db"])
        self.assertEqual(chunk.ports, ["80:80"])
        self.assertEqual(chunk.env_keys, ["PORT", "DB_URL"])
        self.assertEqual(chunk.service_dependencies, {"web": ["db"]})
        self.assertIn("Images: db (postgres:15)", chunk.summary_facts)
        self.assertIn("Builds: web (.)", chunk.summary_facts)

    def test_dockerfile_extraction(self) -> None:
        content = """FROM node:18-alpine
WORKDIR /usr/src/app
COPY package.json pnpm-lock.yaml ./
RUN pnpm install
COPY . .
EXPOSE 3000
CMD ["node", "dist/main.js"]"""
        chunk = Chunk(
            relative_path="Dockerfile",
            chunk_type="file",
            content=content,
        )
        _structured_file_summary(chunk)
        self.assertEqual(chunk.file_type, "dockerfile")
        self.assertEqual(chunk.base_image, "node:18-alpine")
        self.assertEqual(chunk.workdir, "/usr/src/app")
        self.assertEqual(chunk.ports, ["3000"])
        self.assertEqual(chunk.package_manager, "pnpm")
        self.assertEqual(chunk.entrypoints, ['CMD ["node", "dist/main.js"]'])
        self.assertIn("Copied files: package.json, .", chunk.summary_facts)
        self.assertIn("Runtime hints: node", chunk.summary_facts)

    def test_env_example_extraction(self) -> None:
        content = """
# Port config
PORT=8000
# Stripe keys
STRIPE_API_KEY=sk_test_123
STRIPE_WEBHOOK_SECRET=whsec_abc
# Features
ENABLE_BILLING=true
"""
        chunk = Chunk(
            relative_path=".env.example",
            chunk_type="file",
            content=content,
        )
        _structured_file_summary(chunk)
        self.assertEqual(chunk.file_type, "env_example")
        self.assertEqual(chunk.env_keys, ["PORT", "STRIPE_API_KEY", "STRIPE_WEBHOOK_SECRET", "ENABLE_BILLING"])
        self.assertEqual(chunk.feature_flags, ["ENABLE_BILLING"])
        self.assertEqual(chunk.provider_keys, ["STRIPE_API_KEY", "STRIPE_WEBHOOK_SECRET"])

    def test_detect_frameworks_scoped_packages(self) -> None:
        names = [
            "@nestjs/core",
            "@nestjs/common",
            "@qdrant/js-client-rest",
            "@tailwindcss/postcss",
            "@vitejs/plugin-react",
            "drizzle-orm",
        ]
        frameworks = _detect_frameworks(names)
        self.assertIn("NestJS", frameworks)
        self.assertIn("Qdrant", frameworks)
        self.assertIn("Tailwind CSS", frameworks)
        self.assertIn("PostCSS", frameworks)
        self.assertIn("Vite", frameworks)
        self.assertIn("React", frameworks)
        self.assertIn("Drizzle ORM", frameworks)


if __name__ == "__main__":
    unittest.main()
