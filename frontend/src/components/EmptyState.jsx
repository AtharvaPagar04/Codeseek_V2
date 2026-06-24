/**
 * Shown in SessionView when there are no messages yet.
 */
export default function EmptyState({ repoName }) {
  return (
    <div className="flex flex-col items-center justify-center text-center px-8 select-none">
      <div className="text-text-muted font-mono text-lg sm:text-xl mb-4 uppercase tracking-widest">
        {repoName}
      </div>
      <h2 className="text-text-primary text-2xl sm:text-3xl font-medium mb-3 leading-tight">
        Ask anything about{' '}
        <span className="text-text-primary font-mono font-semibold">{repoName}</span>
      </h2>
      <p className="text-text-secondary text-lg max-w-lg">
        Answers are grounded in cited source files from this repository.
      </p>
    </div>
  );
}
