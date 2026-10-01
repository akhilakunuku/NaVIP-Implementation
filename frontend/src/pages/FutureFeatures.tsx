import React from 'react';
import Card from '../components/Card';
import { PHASES } from '../utils/constants';

const FutureFeatures: React.FC = () => {
  const futurePhases = PHASES.filter((p) => p.status === 'locked');

  return (
    <div className="page future-features-page">
      <div className="page-header">
        <h2>Future Navigation Features</h2>
        <p>Planned features for upcoming phases</p>
      </div>

      <Card title="Coming in Future Phases">
        <div className="future-phases">
          {futurePhases.map((phase) => (
            <div key={phase.id} className="future-phase-card">
              <div className="future-phase-header">
                <span className="lock-icon">🔒</span>
                <h4>Phase {phase.id}: {phase.name}</h4>
              </div>
              <p className="future-phase-description">{phase.description}</p>
              <span className="coming-soon-badge">Coming Soon</span>
            </div>
          ))}
        </div>
      </Card>

      <Card title="Phase 5: Navigation Graph">
        <div className="feature-details">
          <p>Planned features:</p>
          <ul>
            <li>Convert map representation into navigation graph</li>
            <li>Nodes</li>
            <li>Edges</li>
            <li>Room connectivity</li>
            <li>Walkable paths</li>
          </ul>
        </div>
      </Card>

      <Card title="Phase 6: Route Planning">
        <div className="feature-details">
          <p>Planned features:</p>
          <ul>
            <li>Select destination</li>
            <li>Calculate route</li>
            <li>Path planning</li>
            <li>Start point → destination</li>
          </ul>
        </div>
      </Card>

      <Card title="Phase 7: Real-Time Navigation">
        <div className="feature-details">
          <p>Planned features:</p>
          <ul>
            <li>Continuous camera input</li>
            <li>Repeated pose estimation</li>
            <li>Current position tracking</li>
            <li>Route progress</li>
            <li>Re-localization</li>
          </ul>
        </div>
      </Card>

      <Card title="Phase 8: Voice Guidance">
        <div className="feature-details">
          <p>Planned features:</p>
          <ul>
            <li>Turn-by-turn instructions</li>
            <li>Audio guidance</li>
            <li>Voice announcements</li>
          </ul>
        </div>
      </Card>

      <Card title="Phase 9: Obstacle / Hazard Assistance">
        <div className="feature-details">
          <p>Planned features:</p>
          <ul>
            <li>Object detection</li>
            <li>Hazard detection</li>
            <li>Distance estimation</li>
            <li>Warning announcements</li>
          </ul>
        </div>
      </Card>

      <Card title="Phase 10: Complete Visually Impaired Navigation System">
        <div className="feature-details">
          <p>Planned integration of:</p>
          <ul>
            <li>Camera</li>
            <li>Pose Estimation</li>
            <li>Map</li>
            <li>Navigation Graph</li>
            <li>Route Planning</li>
            <li>Real-Time Localization</li>
            <li>Hazard Detection</li>
            <li>Voice Guidance</li>
          </ul>
        </div>
      </Card>
    </div>
  );
};

export default FutureFeatures;
