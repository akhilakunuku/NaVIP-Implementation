import { useState } from 'react';
import { poseService } from '../services/poseService';
import type { PosePrediction } from '../types';

interface UsePosePredictionReturn {
  predictPose: (imageFile: File) => Promise<void>;
  prediction: PosePrediction | null;
  loading: boolean;
  error: string | null;
  clearPrediction: () => void;
}

export function usePosePrediction(): UsePosePredictionReturn {
  const [prediction, setPrediction] = useState<PosePrediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const predictPose = async (imageFile: File) => {
    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const result = await poseService.predictPose(imageFile);
      setPrediction(result);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : 'An error occurred during pose prediction'
      );
    } finally {
      setLoading(false);
    }
  };

  const clearPrediction = () => {
    setPrediction(null);
    setError(null);
  };

  return {
    predictPose,
    prediction,
    loading,
    error,
    clearPrediction,
  };
}
