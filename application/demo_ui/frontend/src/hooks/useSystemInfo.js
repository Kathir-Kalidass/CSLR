import { useEffect } from 'react'
import useStore from '../store/useStore'

export default function useSystemInfo() {
  const setSystemInfo = useStore((s) => s.setSystemInfo)

  useEffect(() => {
    let alive = true

    async function fetchInfo() {
      try {
        const res = await fetch('/api/system')
        if (res.ok && alive) {
          setSystemInfo(await res.json())
        }
      } catch { /* retry next interval */ }
    }

    fetchInfo()
    const id = setInterval(fetchInfo, 10000)
    return () => { alive = false; clearInterval(id) }
  }, [setSystemInfo])

  return useStore((s) => s.systemInfo)
}
