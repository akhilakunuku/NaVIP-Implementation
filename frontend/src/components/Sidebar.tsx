import React from 'react';
import { NavLink } from 'react-router-dom';

const Sidebar: React.FC = () => {
  return (
    <nav className="sidebar" role="navigation" aria-label="Main navigation">
      <div className="sidebar-brand">
        <h1>NaVIP</h1>
        <p>AI Indoor Navigation</p>
      </div>
      <ul className="sidebar-nav">
        <li>
          <NavLink to="/" end>
            <span className="nav-icon">📊</span>
            Dashboard
          </NavLink>
        </li>
        <li>
          <NavLink to="/camera">
            <span className="nav-icon">📷</span>
            Camera Input
          </NavLink>
        </li>
        <li>
          <NavLink to="/pose">
            <span className="nav-icon">🎯</span>
            Pose Prediction
          </NavLink>
        </li>
        <li>
          <NavLink to="/localization">
            <span className="nav-icon">📍</span>
            Localization
          </NavLink>
        </li>
        <li>
          <NavLink to="/map">
            <span className="nav-icon">🗺️</span>
            Area 3 Map
          </NavLink>
        </li>
        <li>
          <NavLink to="/rooms">
            <span className="nav-icon">🚪</span>
            Rooms
          </NavLink>
        </li>
        <li>
          <NavLink to="/model">
            <span className="nav-icon">🤖</span>
            Model Info
          </NavLink>
        </li>
        <li>
          <NavLink to="/evaluation">
            <span className="nav-icon">📈</span>
            Evaluation
          </NavLink>
        </li>
        <li>
          <NavLink to="/future">
            <span className="nav-icon">🔮</span>
            Future Features
          </NavLink>
        </li>
      </ul>
      <div className="sidebar-footer">
        <p>Phase 4 Complete</p>
      </div>
    </nav>
  );
};

export default Sidebar;
