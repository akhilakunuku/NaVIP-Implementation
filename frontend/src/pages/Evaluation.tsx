import React, { useState, useEffect } from 'react';
import { evaluationService } from '../services/evaluationService';
import Card from '../components/Card';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorMessage from '../components/ErrorMessage';
import EmptyState from '../components/EmptyState';
import type { EvaluationResult } from '../types';

const Evaluation: React.FC = () => {
  const [results, setResults] = useState<EvaluationResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchResults = async () => {
      setLoading(true);
      setError(null);

      try {
        const result = await evaluationService.getEvaluationResults();
        setResults(result);
      } catch (err) {
        setError(
          err instanceof Error ? err.message : 'Failed to load evaluation results'
        );
      } finally {
        setLoading(false);
      }
    };

    fetchResults();
  }, []);

  if (loading) {
    return <LoadingSpinner message="Loading evaluation results..." />;
  }

  if (error) {
    return (
      <div className="page evaluation-page">
        <div className="page-header">
          <h2>Evaluation / Results</h2>
          <p>Stanford Area 3 test evaluation results</p>
        </div>
        <ErrorMessage message={error} />
        <EmptyState
          icon="📈"
          title="Evaluation data unavailable"
          description="Backend endpoint not available. Evaluation results will load when the backend API is connected."
        />
      </div>
    );
  }

  return (
    <div className="page evaluation-page">
      <div className="page-header">
        <h2>Evaluation / Results</h2>
        <p>Stanford Area 3 test evaluation results</p>
      </div>

      {results && (
        <>
          <Card title="Test Summary">
            <div className="test-summary">
              <div className="summary-item">
                <span className="summary-label">Test Samples:</span>
                <span className="summary-value">{results.testSamples}</span>
              </div>
              <div className="summary-item">
                <span className="summary-label">Successful Predictions:</span>
                <span className="summary-value">{results.successfulPredictions}</span>
              </div>
              <div className="summary-item">
                <span className="summary-label">Failed Predictions:</span>
                <span className="summary-value">{results.failedPredictions}</span>
              </div>
            </div>
          </Card>

          <Card title="Position Error">
            <div className="error-metrics">
              <div className="metric">
                <span className="metric-label">Mean Position Error:</span>
                <span className="metric-value">{results.meanPositionError.toFixed(4)} m</span>
              </div>
              <div className="metric">
                <span className="metric-label">Median Position Error:</span>
                <span className="metric-value">{results.medianPositionError.toFixed(4)} m</span>
              </div>
            </div>
          </Card>

          <Card title="Orientation Error">
            <div className="error-metrics">
              <div className="metric">
                <span className="metric-label">Mean Orientation Error:</span>
                <span className="metric-value">{results.meanOrientationError.toFixed(2)}°</span>
              </div>
              <div className="metric">
                <span className="metric-label">Median Orientation Error:</span>
                <span className="metric-value">{results.medianOrientationError.toFixed(2)}°</span>
              </div>
            </div>
          </Card>

          <Card title="Important Note">
            <p className="evaluation-note">
              These are current experimental results for our Stanford Area 3 implementation.
              These results are NOT from the original NaVIP paper and should not be
              compared directly with published benchmarks.
            </p>
          </Card>
        </>
      )}
    </div>
  );
};

export default Evaluation;
