import { useEffect, useState } from 'react';
import { api } from '../services/api';
import type { GsiStatus } from '../services/api';

const POLL_INTERVAL_MS = 3000;

export function useGsiStatus() {
  const [status, setStatus] = useState<GsiStatus>({
    running: false,
    port: 3000,
    endpoint: '',
    auth_required: false,
    last_error: null,
  });

  useEffect(() => {
    let cancelled = false;

    const poll = async () => {
      try {
        const next = await api.getGsiStatus();
        if (!cancelled) setStatus(next);
      } catch {
        // GSI backend unavailable — keep last known status
      }
    };

    void poll();
    const interval = window.setInterval(poll, POLL_INTERVAL_MS);

    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, []);

  return status;
}
