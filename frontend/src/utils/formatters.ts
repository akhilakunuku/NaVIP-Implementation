export function formatNumber(value: number, decimals: number = 4): string {
  return value.toFixed(decimals);
}

export function formatDegrees(value: number, decimals: number = 2): string {
  return `${value.toFixed(decimals)}°`;
}

export function formatMeters(value: number, decimals: number = 4): string {
  return `${value.toFixed(decimals)} m`;
}
