import { useQuery } from "@tanstack/react-query";

import { api } from "../../shared/api/client";
import type { ModelSummary, Overview } from "../../shared/api/types";
import { REFRESH_MS } from "../../shared/theme/tokens";

export const useOverview = () =>
  useQuery({
    queryKey: ["overview"],
    queryFn: () => api.get<Overview>("/api/overview"),
    refetchInterval: REFRESH_MS,
  });

export const useModels = (kind?: string) =>
  useQuery({
    queryKey: ["models", kind ?? "all"],
    queryFn: () => api.get<ModelSummary[]>(kind ? `/api/models?kind=${kind}` : "/api/models"),
    refetchInterval: REFRESH_MS,
  });
