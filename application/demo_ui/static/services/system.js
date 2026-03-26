export async function fetchSystemInfo() {
  try {
    const resp = await fetch("/api/system");
    if (!resp.ok) {
      return null;
    }
    return await resp.json();
  } catch {
    return null;
  }
}
