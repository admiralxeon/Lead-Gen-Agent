"""Verify the MCP server over HTTP — no subprocess spawning.
First run the server in another terminal:  python leadgen_mcp_server.py --http
Then run this:                             python test_client_http.py
"""

import asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

URL = "http://127.0.0.1:8000/mcp"


async def main():
    async with streamablehttp_client(URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("Discovered tools:", [t.name for t in tools.tools])
            prompts = await session.list_prompts()
            print("Discovered prompts:", [p.name for p in prompts.prompts])
            resources = await session.list_resources()
            print("Discovered resources:", [str(r.uri) for r in resources.resources])

            saved = await session.call_tool(
                "save_lead",
                {
                    "company": "Acme Web Co",
                    "url": "https://acme.example",
                    "tier": "hot",
                    "notes": "needs a redesign",
                },
            )
            print("save_lead ->", saved.content[0].text)

            data = await session.read_resource("leads://all")
            print("leads://all ->", data.contents[0].text.splitlines()[0])


if __name__ == "__main__":
    asyncio.run(main())
