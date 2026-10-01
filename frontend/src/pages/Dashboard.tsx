import React from 'react';
import { useSystemStatus } from '../hooks/SystemStatus';
import Card from '../components/Card';
import PhaseStatus from '../components/PhaseStatus';

const Dashboard: React.FC = () => {
  const status = useSystemStatus();

  return (
    <div className="page dashboard-page">
      <div className="page-header">
        <h2>Dashboard</h2>
        <p>NaVIP — AI-Based Indoor Navigation for Visually Impaired Persons</p>
      </div>

      <div className="dashboard-cards">
        <Card title="Current Dataset">
          <div className="info-row">
            <span className="info-label">Dataset:</span>
            <span className="info-value">Stanford 2D-3D-S</span>
          </div>
        </Card>

        <Card title="Area">
          <div className="info-row">
            <span className="info-label">Area:</span>
            <span className="info-value">Area 3</span>
          </div>
        </Card>

        <Card title="AI Model">
          <div className="info-row">
            <span className="info-label">Model:</span>
            <span className="info-value">ResNet-18 PoseNet</span>
          </div>
        </Card>

        <Card title="Current Phase">
          <div className="info-row">
            <span className="info-label">Phase:</span>
            <span className="info-value">Phase 4 — Map / Spatial Representation</span>
          </div>
        </Card>

        <Card title="Input">
          <div className="info-row">
            <span className="info-label">Input:</span>
            <span className="info-value">RGB Image</span>
          </div>
        </Card>

        <Card title="Output">
          <div className="info-row">
            <span className="info-label">Output:</span>
            <span className="info-value">3D Camera Pose</span>
          </div>
        </Card>
      </div>

      <Card title="System Status">
        <div className="status-grid">
          <div className="status-item">
            <span className="status-label">Backend:</span>
            <span className={`status-badge ${status.backend}`}>
              {status.backend === 'connected' ? 'Connected' : 'Not Connected'}
            </span>
          </div>
          <div className="status-item">
            <span className="status-label">Model:</span>
            <span className={`status-badge ${status.model}`}>
              {status.model === 'available' ? 'Available' : 'Not Available'}
            </span>
          </div>
          <div className="status-item">
            <span className="status-label">Map:</span>
            <span className={`status-badge ${status.map}`}>
              {status.map === 'available' ? 'Available' : 'Not Available'}
            </span>
          </div>
          <div className="status-item">
            <span className="status-label">Dataset:</span>
            <span className="status-badge available">{status.dataset}</span>
          </div>
        </div>
      </Card>

      <PhaseStatus />
    </div>
  );
};

export default Dashboard;
