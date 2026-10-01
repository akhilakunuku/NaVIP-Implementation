import apiClient from '../api/client';
import type { EvaluationResult } from '../types';

export const evaluationService = {
  async getEvaluationResults(): Promise<EvaluationResult> {
    return apiClient.get<EvaluationResult>('/api/evaluation/results');
  },

  async checkEvaluationAvailability(): Promise<boolean> {
    try {
      await apiClient.get('/api/evaluation/results');
      return true;
    } catch {
      return false;
    }
  },
};
