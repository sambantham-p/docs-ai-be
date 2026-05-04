import httpx


http_client = httpx.AsyncClient(
    timeout=httpx.Timeout(8.0, connect=3.0),
    headers={"User-Agent": "Mozilla/5.0"},
)