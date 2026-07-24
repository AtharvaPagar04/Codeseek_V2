# Frontend Architecture

The frontend is a React 18 single-page application built with Vite 5 and Tailwind CSS.

## Application Shell

`frontend/src/App.jsx` renders the root route and owns:

- GitHub connection state.
- Repository-session selection.
- Session and message polling.
- Repository and provider modals.
- Responsive sidebar visibility.

Backend sessions are polled every 60 seconds, or every 15 seconds while any session is indexing.

## State Hooks

| Hook | Responsibility |
|---|---|
| `useSessions` | Normalize sessions and threads, merge polling results, and update messages |
| `useGitHub` | OAuth or token connection, current user, repositories, and logout |
| `useChat` | SSE query lifecycle, cancellation, streamed text, sources, and diagnostics |
| `useRepoGraph` | Graph and retrieval-trace loading, filters, selection, and transformation |
| `useHealth` | Backend health state |

Session state remains in React memory and is refreshed from the backend. Provider model overrides and Graph Assist preferences use browser local storage.

## API Layer

`frontend/src/utils/api.js` centralizes HTTP requests, credential inclusion, error parsing, and server-sent event handling. The default API base is `http://127.0.0.1:8000`; `VITE_API_BASE_URL` overrides it.

## Rendering

Chat answers use `react-markdown` with GitHub-flavored Markdown. Repository and retrieval graphs use `react-force-graph-2d`. Graph code is lazy-loaded from `SessionView`.
