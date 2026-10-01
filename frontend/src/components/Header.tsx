import React from 'react';
import { useSystemStatus } from '../hooks/SystemStatus';

const Header: React.FC = () => {
  const status = useSystemStatus();

  return (
    <header className="header">
      <div className="header-status">
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
      </div>
    </header>
  );
};

export default Header;
