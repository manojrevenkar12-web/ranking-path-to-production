import { useQuery } from "@tanstack/react-query";

import { api } from "../../shared/api/client";
import type { ModelDetail } from "../../shared/api/types";
import { REFRESH_MS } from "../../shared/theme/tokens";

export const useModel = (version: string) =>
  useQuery({
    queryKey: ["model", version],
    queryFn: () => api.get<ModelDetail>(`/api/models/${encodeURIComponent(version)}`),
    refetchInterval: REFRESH_MS,
  });
