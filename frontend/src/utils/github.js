const GH_API = 'https://api.github.com';

const ghHeaders = (token) => ({
  Authorization: `Bearer ${token}`,
  Accept: 'application/vnd.github+json',
  'X-GitHub-Api-Version': '2022-11-28',
});

/**
 * Fetch all repos for the authenticated user (paginates through all pages).
 */
export const fetchUserRepos = async (token) => {
  let page = 1;
  let all = [];
  const params = new URLSearchParams({
    per_page: '100',
    sort: 'updated',
    visibility: 'all',
    affiliation: 'owner,collaborator,organization_member',
  });

  while (true) {
    params.set('page', String(page));
    const res = await fetch(
      `${GH_API}/user/repos?${params.toString()}`,
      { headers: ghHeaders(token) }
    );

    if (!res.ok) {
      throw new Error(`GitHub API error (${res.status})`);
    }

    const batch = await res.json();
    if (!Array.isArray(batch) || batch.length === 0) break;
    all = [...all, ...batch];
    if (batch.length < 100) break;
    page++;
  }

  return all;
};

/**
 * Fetch authenticated GitHub user profile.
 * Returns { login, avatar_url, name }.
 */
export const fetchGithubUser = async (token) => {
  const res = await fetch(`${GH_API}/user`, { headers: ghHeaders(token) });

  if (!res.ok) {
    throw new Error(`Failed to fetch GitHub user (${res.status})`);
  }

  const data = await res.json();
  return { login: data.login, avatar_url: data.avatar_url, name: data.name };
};
