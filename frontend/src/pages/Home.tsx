import React from 'react';
import { Link } from 'react-router-dom';
import './Home.css';
import ParticlesBackground from '../components/ParticlesBackground'; // Adjust path if needed

const Home: React.FC = () => {
  return (
    // Add a wrapper div to handle positioning
    <div className="home-container">
      <ParticlesBackground />
      <div className="main-content">
        {/* Header Section */}
        <header className="home-header">
          <h1>Welcome to the EDA4ALL</h1>
          <p>
            Your all-in-one solution for Automated Exploratory Data Analysis and Machine Learning.
            Streamline your workflow from upload to prediction.
          </p>
        </header>

        {/* Action Cards Grid */}
        <div className="action-grid">
          <div className="action-card">
            <h3>Upload Data</h3>
            <p>
              Effortlessly upload your datasets in CSV, XLS, or XLSX formats to kickstart your analysis.
            </p>
            <Link to="/upload" className="btn btn-primary">
              Upload Dataset
            </Link>
          </div>

          <div className="action-card">
            <h3>Explore Data</h3>
            <p>
              Dive deep into your data with automated EDA, rich visualizations, and insightful summaries.
            </p>
            <Link to="/eda" className="btn btn-primary">
              View EDA
            </Link>
          </div>

          {/* <div className="action-card">
            <h3>Train Models</h3>
            <p>
              Leverage AutoML to train robust models with multiple algorithms and optimizations.
            </p>
            <Link to="/models" className="btn btn-primary">
              Train Models
            </Link>
          </div>

          <div className="action-card">
            <h3>Make Predictions</h3>
            <p>
              Apply your trained models to new data and generate accurate predictions with ease.
            </p>
            <Link to="/predictions" className="btn btn-primary">
              Predict
            </Link>
          </div> */}
        </div>

        {/* Features Section */}
        <section className="section-container">
          <h2 className="section-title">Key Features</h2>
          <div className="feature-grid">
            <div className="feature-item">
              <h3>Automated EDA</h3>
              <ul>
                <li>Statistical summaries and data profiling</li>
                <li>Missing value analysis and imputation</li>
                <li>Distribution analysis and outlier detection</li>
                <li>Advanced correlation analysis</li>
                <li>Interactive visualizations with Plotly</li>
                <li>Automated report generation</li>
              </ul>
            </div>
            <div className="feature-item">
              <h3>Machine Learning Pipeline</h3>
              <ul>
                <li>Auto-detect classification/regression tasks</li>
                <li>Multiple algorithms (Random Forest, LightGBM, XGBoost)</li>
                <li>Automated hyperparameter optimization</li>
                <li>Comprehensive model comparison and selection</li>
                <li>Model explainability with SHAP values</li>
                <li>One-click model deployment</li>
              </ul>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
};

export default Home;