type ViteImportMeta = ImportMeta & {
  env?: {
    VITE_API_BASE_URL?: string;
  };
};

const configuredBase = (import.meta as ViteImportMeta).env?.VITE_API_BASE_URL?.trim();

export const API_BASE =
  configuredBase ||
  `${window.location.protocol}//${window.location.hostname}:8000`;

export function apiUrl(path: string): string {
  return `${API_BASE}${path.startsWith("/") ? path : `/${path}`}`;
}
