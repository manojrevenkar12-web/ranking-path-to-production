const TOKEN_KEY = "ptp.token";
let memoryToken: string | null = null;

/** The token for the one write (rollback). Kept for this tab only. */
export const session = {
  token(): string | null {
    try {
      return sessionStorage.getItem(TOKEN_KEY) ?? memoryToken;
    } catch {
      return memoryToken;
    }
  },
  setToken(t: string) {
    memoryToken = t.trim();
    try {
      sessionStorage.setItem(TOKEN_KEY, memoryToken);
    } catch {
      /* private mode: kept in memory */
    }
  },
  reset() {
    memoryToken = null;
    try {
      sessionStorage.removeItem(TOKEN_KEY);
    } catch {
      /* nothing kept */
    }
  },
};

/** An API failure with what the API said and whether trying again may help. */
export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly title: string,
    readonly detail?: string,
  ) {
    super(detail ? `${title}: ${detail}` : title);
  }

  get retryable() {
    return this.status === 0 || this.status === 429 || this.status >= 500;
  }
}

async function send(method: string, path: string, json?: unknown): Promise<Response> {
  const headers: Record<string, string> = { accept: "application/json" };
  const t = session.token();
  if (t) headers["authorization"] = `Bearer ${t}`;
  if (json !== undefined) headers["content-type"] = "application/json";
  let response: Response;
  try {
    response = await fetch(path, {
      method,
      headers,
      body: json === undefined ? undefined : JSON.stringify(json),
    });
  } catch {
    throw new ApiError(0, "The registry API cannot be reached", "Check that `ptp serve` is running.");
  }
  if (!response.ok) {
    let detail: string | undefined;
    try {
      const body = (await response.json()) as { detail?: unknown };
      detail = typeof body.detail === "string" ? body.detail : undefined;
    } catch {
      /* no body */
    }
    const title =
      response.status === 401
        ? "This needs a token"
        : response.status === 404
          ? "Not found"
          : `The API answered ${response.status}`;
    throw new ApiError(response.status, title, detail);
  }
  return response;
}

export const api = {
  get: async <T>(path: string) => (await (await send("GET", path)).json()) as T,
  post: async <T>(path: string, json: unknown) => (await (await send("POST", path, json)).json()) as T,
};
