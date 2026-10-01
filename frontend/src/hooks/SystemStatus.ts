import { useState, useEffect } from 'react';
import apiClient from '../api/client';
import { poseService } from '../services/poseService';
import { mapService } from '../services/mapService';
import type { SystemStatus } from '../types';

export function useSystemStatus() {
  const [status, setStatus] = useState<SystemStatus>({
    backend: 'checking',
    model: 'checking',
    map: 'checking',
    dataset: 'Stanford 2D-3D-S Area 3',
  });

  useEffect(() => {
    const checkStatus = async () => {
      const isBackendConnected = await apiClient.checkHealth();

      if (!isBackendConnected) {
        setStatus({
          backend: 'not_connected',
          model: 'not_available',
          map: 'not_available',
          dataset: 'Stanford 2D-3D-S Area 3',
        });
        return;
      }

      const isModelAvailable = await poseService.checkModelAvailability();
      const isMapAvailable = await mapService.checkMapAvailability();

      setStatus({
        backend: 'connected',
        model: isModelAvailable ? 'available' : 'not_available',
        map: isMapAvailable ? 'available' : 'not_available',
        dataset: 'Stanford 2D-3D-S Area 3',
      });
    };

    checkStatus();
  }, []);

  return status;
}

