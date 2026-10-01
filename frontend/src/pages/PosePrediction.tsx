import React from 'react';
import { usePosePrediction } from '../hooks/PosePrediction';
import Card from '../components/Card';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorMessage from '../components/ErrorMessage';
import EmptyState from '../components/EmptyState';

const PosePrediction: React.FC = () => {
  const { prediction, loading, error, clearPrediction } = usePosePrediction();

  return (
    <div className="page pose-prediction-page">
      <div className="page-header">
        <h2>Pose Prediction</h2>
        <p>Camera pose estimation results</p>
      </div>

      {loading && <LoadingSpinner message="Predicting camera pose..." />}

      {error && <ErrorMessage message={error} onDismiss={clearPrediction} />}

      {prediction ? (
        <div className="pose-results">
          <Card title="Predicted Camera Position">
            <div className="pose-display">
              <div className="coordinate">
                <span className="coordinate-label">X</span>
                <span className="coordinate-value">{prediction.position.x.toFixed(4)}</span>
                <span className="coordinate-unit">m</span>
              </div>
              <div className="coordinate">
                <span className="coordinate-label">Y</span>
                <span className="coordinate-value">{prediction.position.y.toFixed(4)}</span>
                <span className="coordinate-unit">m</span>
              </div>
              <div className="coordinate">
                <span className="coordinate-label">Z</span>
                <span className="coordinate-value">{prediction.position.z.toFixed(4)}</span>
                <span className="coordinate-unit">m</span>
              </div>
            </div>
          </Card>

          <Card title="Predicted Orientation">
            <div className="pose-display">
              <div className="coordinate">
                <span className="coordinate-label">qw</span>
                <span className="coordinate-value">{prediction.orientation.qw.toFixed(4)}</span>
              </div>
              <div className="coordinate">
                <span className="coordinate-label">qx</span>
                <span className="coordinate-value">{prediction.orientation.qx.toFixed(4)}</span>
              </div>
              <div className="coordinate">
                <span className="coordinate-label">qy</span>
                <span className="coordinate-value">{prediction.orientation.qy.toFixed(4)}</span>
              </div>
              <div className="coordinate">
                <span className="coordinate-label">qz</span>
                <span className="coordinate-value">{prediction.orientation.qz.toFixed(4)}</span>
              </div>
            </div>
          </Card>

          {prediction.groundTruthPosition && (
            <Card title="Ground Truth Position">
              <div className="pose-display">
                <div className="coordinate">
                  <span className="coordinate-label">X</span>
                  <span className="coordinate-value">{prediction.groundTruthPosition.x.toFixed(4)}</span>
                  <span className="coordinate-unit">m</span>
                </div>
                <div className="coordinate">
                  <span className="coordinate-label">Y</span>
                  <span className="coordinate-value">{prediction.groundTruthPosition.y.toFixed(4)}</span>
                  <span className="coordinate-unit">m</span>
                </div>
                <div className="coordinate">
                  <span className="coordinate-label">Z</span>
                  <span className="coordinate-value">{prediction.groundTruthPosition.z.toFixed(4)}</span>
                  <span className="coordinate-unit">m</span>
                </div>
              </div>
            </Card>
          )}

          {prediction.groundTruthOrientation && (
            <Card title="Ground Truth Orientation">
              <div className="pose-display">
                <div className="coordinate">
                  <span className="coordinate-label">qw</span>
                  <span className="coordinate-value">{prediction.groundTruthOrientation.qw.toFixed(4)}</span>
                </div>
                <div className="coordinate">
                  <span className="coordinate-label">qx</span>
                  <span className="coordinate-value">{prediction.groundTruthOrientation.qx.toFixed(4)}</span>
                </div>
                <div className="coordinate">
                  <span className="coordinate-label">qy</span>
                  <span className="coordinate-value">{prediction.groundTruthOrientation.qy.toFixed(4)}</span>
                </div>
                <div className="coordinate">
                  <span className="coordinate-label">qz</span>
                  <span className="coordinate-value">{prediction.groundTruthOrientation.qz.toFixed(4)}</span>
                </div>
              </div>
            </Card>
          )}

          {prediction.positionError !== undefined && prediction.orientationError !== undefined && (
            <Card title="Prediction Errors">
              <div className="error-display">
                <div className="error-item">
                  <span className="error-label">Position Error:</span>
                  <span className="error-value">{prediction.positionError.toFixed(4)} m</span>
                </div>
                <div className="error-item">
                  <span className="error-label">Orientation Error:</span>
                  <span className="error-value">{prediction.orientationError.toFixed(2)}°</span>
                </div>
              </div>
            </Card>
          )}
        </div>
      ) : (
        !loading && !error && (
          <EmptyState
            icon="🎯"
            title="No prediction available"
            description="Upload an image from the Camera Input page to get pose predictions"
          />
        )
      )}
    </div>
  );
};

export default PosePrediction;
