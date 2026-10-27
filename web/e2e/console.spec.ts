import { expect, test } from "@playwright/test";

// Against `ptp serve` after `ptpsim run scenarios/dev/smoke.yaml` (see .github/workflows/ci.yml, job e2e).

test("the path shows both surfaces and every recent version's rail", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { level: 1, name: "Path to production" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Search" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Homepage" })).toBeVisible();
  const rails = page.getByRole("list", { name: /^Path of / });
  await expect(rails.first()).toBeVisible();
  expect(await rails.count()).toBeGreaterThan(1);
});

test("a version's page explains its gates", async ({ page }) => {
  await page.goto("/models");
  await page
    .getByRole("link", { name: /^r\d{4}$/ })
    .first()
    .click();
  await expect(page.getByRole("heading", { level: 1, name: /^r\d{4}$/ })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Gates" })).toBeVisible();
  const gate = page.getByRole("button", { expanded: false }).first();
  await gate.click();
  await expect(page.getByText(/^Gate /).first()).toBeVisible();
});

test("runs link to their gates and their event", async ({ page }) => {
  await page.goto("/runs");
  await page
    .getByRole("link", { name: /^ct-ranker-d\d\d$/ })
    .first()
    .click();
  await expect(page.getByRole("heading", { name: "Event" })).toBeVisible();
});

test("the evidence page shows the season against its truth", async ({ page }) => {
  await page.goto("/evidence");
  await expect(page.getByText(/harmful releases reached production/).first()).toBeVisible();
});

test("the on-call can roll search back with a token", async ({ page }, info) => {
  test.skip(info.project.name !== "desktop", "one rollback per run: it changes the registry");
  await page.goto("/routing");
  const panel = page.locator("section", { has: page.getByRole("heading", { name: "Search" }) });
  const button = panel.getByRole("button", { name: "Roll back production" });
  test.skip((await button.count()) === 0, "nothing to roll back to in this season");
  await button.click();
  await panel.getByLabel("Why").fill("e2e: checking the rollback path");
  await panel.getByLabel("API token").fill("e2e-token");
  await panel.getByRole("button", { name: "Roll back search" }).click();
  await expect(page.getByText(/rollback: e2e: checking the rollback path/)).toBeVisible();
});
