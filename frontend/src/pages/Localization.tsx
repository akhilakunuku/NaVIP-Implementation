import React from 'react';
import { usePosePrediction } from '../hooks/PosePrediction';
import Card from '../components/Card';
import EmptyState from '../components/EmptyState';

const Localization: React.FC = () => {
  const { prediction } = usePosePrediction();

  return (
    <div className="page localization-page">
      <div className="page-header">
        <h2>Localization</h2>
        <p>Camera position visualization on Area 3 map</p>
      </div>

      {prediction ? (
        <div className="localization-content">
          <Card title="Uploaded Image">
            <div className="image-container">
              <p className="placeholder-text">Image preview would appear here</p>
            </div>
          </Card>

          <Card title="Predicted Position">
            <div className="position-info">
              <div className="position-item">
                <span className="position-label">X:</span>
                <span className="position-value">{prediction.position.x.toFixed(4)} m</span>
              </div>
              <div className="position-item">
                <span className="position-label">Y:</span>
                <span className="position-value">{prediction.position.y.toFixed(4)} m</span>
              </div>
              <div className="position-item">
                <span className="position-label">Z:</span>
                <span className="position-value">{prediction.position.z.toFixed(4)} m</span>
              </div>
            </div>
          </Card>

          <Card title="Area 3 Derived Navigation Map">
            <div className="map-container">
              <div className="map-placeholder">
                <div className="map-grid">
                  <div className="map-room office">
                    <span>Office</span>
                  </div>
                  <div className="map-room hallway">
                    <span>Hallway</span>
                  </div>
                  <div className="map-room office2">
                    <span>Office</span>
                  </div>
                  <div className="map-room lounge">
                    <span>Lounge</span>
                  </div>
                  <div className="map-marker" title="Predicted Position">
                    <span className="marker-icon">●</span>
                    <span className="marker-label">YOU</span>
                  </div>
                </div>
                <p className="map-note">
                  Area 3 Derived Navigation Map — Not an official Stanford floor plan
                </p>
              </div>
            </div>
          </Card>
        </div>
      ) : (
        <EmptyState
          icon="📍"
          title="No localization data"
          description="Upload an image and get a pose prediction to see localization"
        />
      )}
    </div>
  );
};

export default Localization;
