import React from 'react';
import { Link, useLocation } from 'react-router-dom';

const Navbar: React.FC = () => {
  const location = useLocation();

  const isActive = (path: string) => {
    return location.pathname === path || location.pathname.startsWith(path);
  };

  return (
    <nav className="navbar">
      <div className="navbar-content">
        <Link to="/" className="navbar-brand">
          🤖 AI EDA Platform
        </Link>
        <ul className="navbar-nav">
          <li>
            <Link to="/" className={isActive('/') && location.pathname === '/' ? 'active' : ''}>
              Home
            </Link>
          </li>
          <li>
            <Link to="/upload" className={isActive('/upload') ? 'active' : ''}>
              Upload Data
            </Link>
          </li>
          <li>
            <Link to="/eda" className={isActive('/eda') ? 'active' : ''}>
              EDA
            </Link>
          </li>
          <li>
            <Link to="/models" className={isActive('/models') ? 'active' : ''}>
              Models
            </Link>
          </li>
          <li>
            <Link to="/predictions" className={isActive('/predictions') ? 'active' : ''}>
              Predictions
            </Link>
          </li>
        </ul>
      </div>
    </nav>
  );
};

export default Navbar;
