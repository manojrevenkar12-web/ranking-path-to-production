import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Link, Route, Routes } from "react-router-dom";

import { EvidencePage } from "../features/evidence/EvidencePage";
import { ModelPage } from "../features/model/ModelPage";
import { ModelsPage } from "../features/path/ModelsPage";
import { PathPage } from "../features/path/PathPage";
import { RoutingPage } from "../features/routing/RoutingPage";
import { RunPage, RunsPage } from "../features/runs/RunsPage";
import { StandardsPage } from "../features/standards/StandardsPage";
import { EmptyState } from "../shared/ui";
import { Shell } from "./Shell";

export function makeQueryClient() {
  return new QueryClient({
    defaultOptions: {
      queries: {
        retry: (count, error) => count < 2 && (error as { retryable?: boolean }).retryable !== false,
        staleTime: 5_000,
      },
    },
  });
}

export function AppRoutes() {
  return (
    <Shell>
      <Routes>
        <Route path="/" element={<PathPage />} />
        <Route path="/runs" element={<RunsPage />} />
        <Route path="/runs/:runId" element={<RunPage />} />
        <Route path="/models" element={<ModelsPage />} />
        <Route path="/models/:version" element={<ModelPage />} />
        <Route path="/routing" element={<RoutingPage />} />
        <Route path="/standards" element={<StandardsPage />} />
        <Route path="/evidence" element={<EvidencePage />} />
        <Route
          path="*"
          element={
            <EmptyState
              title="There is nothing at this address"
              action={<Link to="/">Back to the path</Link>}
            >
              The page may have moved, or the link is mistyped.
            </EmptyState>
          }
        />
      </Routes>
    </Shell>
  );
}

export function App() {
  return (
    <QueryClientProvider client={makeQueryClient()}>
      <BrowserRouter>
        <AppRoutes />
      </BrowserRouter>
    </QueryClientProvider>
  );
}
