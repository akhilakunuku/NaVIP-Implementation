import apiClient from '../api/client';
import type { PosePrediction } from '../types';

export interface PredictPoseResponse {
  position: {
    x: number;
    y: number;
    z: number;
  };
  orientation: {
    qw: number;
    qx: number;
    qy: number;
    qz: number;
  };
  positionError?: number;
  orientationError?: number;
  groundTruthPosition?: {
    x: number;
    y: number;
    z: number;
  };
  groundTruthOrientation?: {
    qw: number;
    qx: number;
    qy: number;
    qz: number;
  };
}

export const poseService = {
  async predictPose(imageFile: File): Promise<PosePrediction> {
    const formData = new FormData();
    formData.append('image', imageFile);

    const response = await apiClient.postFormData<PredictPoseResponse>(
      '/api/pose/predict',
      formData
    );

    return {
      position: response.position,
      orientation: response.orientation,
      positionError: response.positionError,
      orientationError: response.orientationError,
      groundTruthPosition: response.groundTruthPosition,
      groundTruthOrientation: response.groundTruthOrientation,
    };
  },

  async checkModelAvailability(): Promise<boolean> {
    try {
      await apiClient.get('/api/model/status');
      return true;
    } catch {
      return false;
    }
  },
};
