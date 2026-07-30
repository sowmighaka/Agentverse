from typing import Any

import httpx
from bs4 import BeautifulSoup


FETCH_TIMEOUT = 15.0


class FetchError(RuntimeError):
    """Raised when a required external source cannot be fetched."""


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "nav", "footer"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


async def fetch_page_text(url: str) -> str:
    try:
        async with httpx.AsyncClient(
            timeout=FETCH_TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": "AgentverseTailor/1.0"},
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise FetchError(f"Could not fetch {url}: {exc}") from exc

    text = html_to_text(response.text)
    if not text:
        raise FetchError(f"Fetched {url}, but no readable text was found")
    return text[:50000]


async def fetch_github_org_repos(org: str) -> list[dict[str, Any]]:
    url = f"https://api.github.com/orgs/{org}/repos"
    params = {"sort": "updated", "per_page": 20}
    try:
        async with httpx.AsyncClient(timeout=FETCH_TIMEOUT, headers={"User-Agent": "AgentverseTailor/1.0"}) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            repos = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise FetchError(f"Could not fetch GitHub organization '{org}': {exc}") from exc

    if not isinstance(repos, list):
        raise FetchError(f"GitHub organization '{org}' did not return a repository list")

    return [
        {
            "name": repo.get("name"),
            "language": repo.get("language"),
            "topics": repo.get("topics") or [],
            "description": repo.get("description"),
            "updated_at": repo.get("updated_at"),
        }
        for repo in repos[:20]
        if isinstance(repo, dict)
    ]
