export const OPENAI_COMPATIBLE_EMBEDDING_MODELS = {
  "text-embedding-3-small": [512, 1536],
  "text-embedding-3-small": [512, 1536],
  "text-embedding-3-large": [256, 1024, 3072],
  "text-embedding-3-large": [256, 1024, 3072],
  "openai/text-embedding-ada-002": [1536],
  "text-embedding-ada-002": [1536],
  "google/gemini-embedding-001": [],
  "google/gemini-embedding-2-preview": [],
  "google/text-embedding-004": [],
};

export function validateEmbeddingDimensions(mode, model, dims) {
  if (mode !== 'api') return null;
  const modelName = model.trim();
  const allowed = OPENAI_COMPATIBLE_EMBEDDING_MODELS[modelName];
  if (!allowed) return null;
  
  const d = dims ? parseInt(dims, 10) : 0;
  if (d !== 0 && !allowed.includes(d)) {
    let suggestion = 'Auto';
    if (allowed.length > 0) {
      const last = allowed[allowed.length - 1];
      const rest = allowed.slice(0, -1);
      if (rest.length > 0) {
         suggestion = `Auto, ${rest.join(', ')}, or ${last}`;
      } else {
         suggestion = `Auto or ${last}`;
      }
    }
    return `${dims} is not valid for ${modelName}. Use ${suggestion}.`;
  }
  return null;
}
