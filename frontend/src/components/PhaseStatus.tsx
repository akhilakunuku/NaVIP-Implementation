import React from 'react';
import { PHASES } from '../utils/constants';

const PhaseStatus: React.FC = () => {
  return (
    <div className="phase-status">
      <h3>Project Progress</h3>
      <div className="phase-list">
        {PHASES.map((phase) => (
          <div key={phase.id} className={`phase-item ${phase.status}`}>
            <span className="phase-icon">
              {phase.status === 'completed' ? '✓' : '🔒'}
            </span>
            <div className="phase-info">
              <span className="phase-name">Phase {phase.id}: {phase.name}</span>
              <span className="phase-description">{phase.description}</span>
            </div>
            <span className={`phase-badge ${phase.status}`}>
              {phase.status === 'completed' ? 'Completed' : 'Coming Soon'}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default PhaseStatus;
