import os
import json
from fastmcp import FastMCP

BASE = os.path.dirname(__file__)
DOMAIN_DIR = os.path.abspath(os.path.join(BASE, "..", "domain_knowledge"))

server = FastMCP("knowledge-servers")
def _load_domain(domain: str):
    path = os.path.join(DOMAIN_DIR, f"{domain}_knowledge.json")
    if not os.path.exists(path):
        # fallback: allow plain "diabetes.json"
        alt = os.path.join(DOMAIN_DIR, f"{domain}.json")
        path = alt if os.path.exists(alt) else None
    if not path:
        return {"domain": domain, "arguments": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@server.tool()
def retrieve_knowledge(query: str, domain: str = "titanic") -> dict:
    data = _load_domain(domain)
    matches = []
    for arg in data.get("arguments", []):
        features = [s.lower() for s in arg.get("features", [])]
        if query.lower() in features or query.lower() in arg.get("text", "").lower():
            matches.append(arg.get("text", ""))
    return {"domain": domain, "query": query, "results": matches[:10]}

if __name__ == "__main__":
    server.run()
