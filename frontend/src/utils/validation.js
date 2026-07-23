export const DEFAULT_API_EMBEDDING_MODEL = 'text-embedding-3-small';
export const DEFAULT_LOCAL_EMBEDDING_MODEL = 'nomic-embed-text:latest';

export function defaultEmbeddingModelForMode(mode) {
  return mode === 'local' ? DEFAULT_LOCAL_EMBEDDING_MODEL : DEFAULT_API_EMBEDDING_MODEL;
}
