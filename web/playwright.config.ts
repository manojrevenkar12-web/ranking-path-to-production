import { defineConfig, devices } from "@playwright/test";

// Runs against `ptp serve` serving the built console (see .github/workflows/ci.yml):
// after `ptpsim run scenarios/dev/smoke.yaml` and `npm run build`.
export default defineConfig({
  testDir: "e2e",
  timeout: 60_000,
  retries: process.env["CI"] ? 1 : 0,
  workers: 1,
  reporter: process.env["CI"] ? [["list"], ["html", { open: "never" }]] : "list",
  use: {
    baseURL: process.env["E2E_BASE_URL"] ?? "http://localhost:8000",
    trace: "retain-on-failure",
    locale: "en-GB",
    // A machine with a different Chromium than this Playwright expects can point at it.
    launchOptions: process.env["PW_CHROMIUM_PATH"] ? { executablePath: process.env["PW_CHROMIUM_PATH"] } : {},
  },
  projects: [
    { name: "desktop", use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 960 } } },
    { name: "phone", use: { ...devices["Pixel 7"] } },
  ],
});
