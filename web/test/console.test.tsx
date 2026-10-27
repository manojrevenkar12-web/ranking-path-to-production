import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { PathRail } from "../src/features/path/PathRail";
import type {
  ModelDetail,
  ModelSummary,
  Overview,
  Run,
  Season,
  StandardsReport,
} from "../src/shared/api/types";
import { session } from "../src/shared/api/client";
import fixtures from "./fixtures/season.json";
import { renderAt } from "./render";

const F = fixtures as unknown as Record<string, unknown>;
const overview = F["overview"] as Overview;
const models = F["models"] as ModelSummary[];
const runs = F["runs"] as Run[];

const stoppedVersion = models.find(
  (m) => m.path?.some((p) => p.state === "stopped") && F[`models/${m.version}`],
);
const promotedVersion = models.find(
  (m) => m.status === "production" && m.path?.some((p) => p.station === "canary" && p.state === "passed"),
);

beforeEach(() => session.reset());

describe("the path rail", () => {
  it("names all six stations and where the version stopped", () => {
    render(
      <PathRail
        version="r0009"
        path={[
          { station: "data", state: "passed", gates: [] },
          { station: "model", state: "passed", gates: [] },
          { station: "export", state: "stopped", gates: ["export.target_compat"] },
          { station: "staging", state: "skipped", gates: [] },
          { station: "canary", state: "skipped", gates: [] },
          { station: "production", state: "skipped", gates: [] },
        ]}
      />,
    );
    const rail = screen.getByRole("list", { name: "Path of r0009" });
    const items = within(rail).getAllByRole("listitem");
    expect(items.map((li) => li.querySelector("span:nth-child(2)")?.textContent)).toEqual([
      "Data",
      "Offline",
      "Export",
      "Staging",
      "Canary",
      "Production",
    ]);
    expect(within(items[2]!).getByText("Stopped")).toBeInTheDocument();
    expect(within(items[2]!).getByText("Loads in the serving runtime")).toBeInTheDocument();
  });
});

describe("the path page", () => {
  it("shows a skeleton, then both surfaces, what serves them and the versions on the path", async () => {
    renderAt("/", F);
    expect(screen.getByRole("status", { name: "Loading the path" })).toBeInTheDocument();
    expect(await screen.findByRole("heading", { name: "Search" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Homepage" })).toBeInTheDocument();
    const search = overview.routing.find((r) => r.surface === "search")!;
    expect(
      screen.getAllByRole("link", { name: search.production.replace("ranker_", "") }).length,
    ).toBeGreaterThan(0);
    expect(screen.getAllByRole("list", { name: /^Path of / })).toHaveLength(overview.recent.length);
  });

  it("says what failed and retries", async () => {
    let fail = true;
    renderAt("/", F, (url) =>
      url.includes("overview") && fail
        ? { status: 503, body: { detail: "registry is restarting" } }
        : undefined,
    );
    expect(await screen.findByRole("alert")).toHaveTextContent("registry is restarting");
    fail = false;
    await userEvent.click(screen.getByRole("button", { name: "Try again" }));
    expect(await screen.findByRole("heading", { name: "Search" })).toBeInTheDocument();
  });

  it("explains how to start when nothing has run", async () => {
    const empty = { ...overview, recent: [], runs: [], routing: [], production: {}, open_deployments: [] };
    renderAt("/", { ...F, overview: empty });
    expect(await screen.findByRole("heading", { name: "Nothing on the path yet" })).toBeInTheDocument();
  });
});

describe("a model version", () => {
  it("shows where it stopped, its gates by stage, and the evidence behind one", async () => {
    expect(stoppedVersion).toBeDefined();
    const v = stoppedVersion!.version;
    const detail = F[`models/${v}`] as ModelDetail;
    renderAt(`/models/${v}`, F);
    expect(await screen.findByRole("heading", { level: 1, name: v })).toBeInTheDocument();
    expect(screen.getByRole("list", { name: `Path of ${v}` })).toBeInTheDocument();
    const blocking = detail.gates.find((g) => g.blocking)!;
    const row = screen.getAllByRole("button", {
      name: new RegExp(blocking.summary.slice(0, 20).replace(/[.*+?^${}()|[\]\\]/g, "\\$&")),
    })[0]!;
    expect(row).toHaveAttribute("aria-expanded", "false");
    await userEvent.click(row);
    expect(row).toHaveAttribute("aria-expanded", "true");
    expect(
      await screen.findByText(new RegExp(`Gate ${blocking.gate.replace(".", "\\.")}`)),
    ).toBeInTheDocument();
  });

  it("plots the offline comparison per market", async () => {
    const withMetrics = models.find(
      (m) =>
        (F[`models/${m.version}`] as ModelDetail | undefined)?.model.metrics?.markets && m.kind === "ranker",
    )!;
    renderAt(`/models/${withMetrics.version}`, F);
    expect(await screen.findByRole("img", { name: /^All markets:/ })).toBeInTheDocument();
    for (const m of ["FR", "DE", "PL", "LT"])
      expect(screen.getByRole("img", { name: new RegExp(`^${m}:`) })).toBeInTheDocument();
  });

  it("shows the canary decision and the deployment steps of a promoted version", async () => {
    expect(promotedVersion).toBeDefined();
    renderAt(`/models/${promotedVersion!.version}`, F);
    expect(await screen.findByRole("heading", { name: "Deployments" })).toBeInTheDocument();
    expect(screen.getAllByText(/deploy prod/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/promote/i).length).toBeGreaterThan(0);
  });

  it("says when a version does not exist", async () => {
    renderAt("/models/r9999", F);
    expect(await screen.findByRole("alert")).toHaveTextContent("Not found");
  });
});

describe("pipeline runs", () => {
  it("lists runs with their outcome and the gate that stopped them", async () => {
    renderAt("/runs", F);
    const table = await screen.findByRole("table");
    expect(within(table).getAllByRole("row")).toHaveLength(runs.length + 1);
    const blocked = runs.find((r) => r.blocking.length > 0)!;
    const row = within(table).getByRole("link", { name: blocked.run_id }).closest("tr")!;
    expect(within(row).getByText("Blocked")).toBeInTheDocument();
  });
});

describe("routing", () => {
  it("rolls production back with a reason and a token, and says when the token is refused", async () => {
    const posts: RequestInit[] = [];
    let accept = false;
    renderAt("/routing", F, (_url, init) => {
      if (init?.method === "POST") {
        posts.push(init);
        return accept
          ? { status: 200, body: { production: "x" } }
          : { status: 401, body: { detail: "bad token" } };
      }
      return undefined;
    });
    const search = await screen.findByRole("heading", { name: "Search" });
    const panel = search.closest("section")!;
    await userEvent.click(within(panel).getByRole("button", { name: "Roll back production" }));
    const submit = within(panel).getByRole("button", { name: "Roll back search" });
    expect(submit).toBeDisabled();
    await userEvent.type(within(panel).getByLabelText("Why"), "conversion dropped in FR");
    await userEvent.type(within(panel).getByLabelText("API token"), "wrong");
    await userEvent.click(submit);
    expect(await within(panel).findByRole("alert")).toHaveTextContent("That token was not accepted.");
    expect((posts[0]!.headers as Record<string, string>)["authorization"]).toBe("Bearer wrong");
    accept = true;
    await userEvent.clear(within(panel).getByLabelText("API token"));
    await userEvent.type(within(panel).getByLabelText("API token"), "right");
    await userEvent.click(within(panel).getByRole("button", { name: "Roll back search" }));
    await waitFor(() => expect(within(panel).queryByLabelText("Why")).not.toBeInTheDocument());
    expect(JSON.parse(String(posts[1]!.body))).toEqual({ reason: "conversion dropped in FR" });
  });
});

describe("standards", () => {
  it("shows each production model against every standard", async () => {
    const report = F["standards"] as StandardsReport;
    renderAt("/standards", F);
    const table = await screen.findByRole("table");
    expect(within(table).getAllByRole("columnheader")).toHaveLength(report.standards.length + 1);
    const prod = report.models.find((m) => m.status === "production")!;
    expect(within(table).getByRole("link", { name: prod.version })).toBeInTheDocument();
  });
});

describe("evidence", () => {
  it("puts each release's real effect next to what the platform did", async () => {
    const seasons = F["seasons"] as Season[];
    renderAt("/evidence", F);
    expect(await screen.findByRole("heading", { name: seasons[0]!.scenario })).toBeInTheDocument();
    expect(screen.getAllByRole("table")).toHaveLength(seasons.length);
    expect(screen.getAllByText(/harmful releases reached production/).length).toBe(seasons.length);
  });

  it("filters versions by model", async () => {
    const { calls } = renderAt("/models", F);
    await screen.findAllByRole("list", { name: /^Path of / });
    await userEvent.click(screen.getByRole("radio", { name: "Homepage two-tower" }));
    await waitFor(() => expect(calls.some((c) => c.url === "/api/models?kind=tower")).toBe(true));
  });

  it("has a not-found page", async () => {
    renderAt("/nowhere", F);
    expect(
      await screen.findByRole("heading", { name: "There is nothing at this address" }),
    ).toBeInTheDocument();
  });
});
