import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

import { AppRoutes } from "../src/app/App";

type Handler = (url: string, init?: RequestInit) => { status: number; body: unknown } | undefined;

/** The console at `path`, with fetch answered from captured API responses (test/fixtures, from a dev season). */
export function renderAt(path: string, fixtures: Record<string, unknown>, handler?: Handler) {
  const calls: { url: string; init?: RequestInit }[] = [];
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input.toString();
    calls.push({ url, init });
    const custom = handler?.(url, init);
    if (custom)
      return new Response(JSON.stringify(custom.body), {
        status: custom.status,
        headers: { "content-type": "application/json" },
      });
    const key = url.replace(/^\/api\//, "").replace(/\?.*$/, (q) => q.replace("?", "_").replace("=", "-"));
    if (key in fixtures)
      return new Response(JSON.stringify(fixtures[key]), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    return new Response(JSON.stringify({ detail: `no fixture for ${url}` }), { status: 404 });
  });
  vi.stubGlobal("fetch", fetchMock);
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const utils = render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[path]}>
        <AppRoutes />
      </MemoryRouter>
    </QueryClientProvider>,
  );
  return { ...utils, calls, fetchMock };
}
