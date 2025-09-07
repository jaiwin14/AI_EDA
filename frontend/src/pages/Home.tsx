import React from 'react';
import { Link } from 'react-router-dom';

const Home: React.FC = () => {
  return (
    <div className="text-center">
      <div className="card">
        <div className="card-header">
          <h1 className="text-2xl font-bold mb-4">Welcome to AI EDA Platform</h1>
          <p className="text-gray-600">
            Automated Exploratory Data Analysis and Machine Learning Pipeline
          </p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mt-6">
          <div className="card">
            <div className="text-center">
              <div className="text-4xl mb-4">📊</div>
              <h3 className="font-semibold mb-2">Upload Data</h3>
              <p className="text-sm text-gray-600 mb-4">
                Upload CSV, XLS, or XLSX files to get started
              </p>
              <Link to="/upload" className="btn btn-primary btn-sm">
                Upload Dataset
              </Link>
            </div>
          </div>

          <div className="card">
            <div className="text-center">
              <div className="text-4xl mb-4">🔍</div>
              <h3 className="font-semibold mb-2">Explore Data</h3>
              <p className="text-sm text-gray-600 mb-4">
                Automated EDA with visualizations and insights
              </p>
              <Link to="/eda" className="btn btn-secondary btn-sm">
                View EDA
              </Link>
            </div>
          </div>

          <div className="card">
            <div className="text-center">
              <div className="text-4xl mb-4">🤖</div>
              <h3 className="font-semibold mb-2">Train Models</h3>
              <p className="text-sm text-gray-600 mb-4">
                Automated ML with multiple algorithms
              </p>
              <Link to="/models" className="btn btn-secondary btn-sm">
                Train Models
              </Link>
            </div>
          </div>

          <div className="card">
            <div className="text-center">
              <div className="text-4xl mb-4">🎯</div>
              <h3 className="font-semibold mb-2">Make Predictions</h3>
              <p className="text-sm text-gray-600 mb-4">
                Use trained models for predictions
              </p>
              <Link to="/predictions" className="btn btn-secondary btn-sm">
                Predict
              </Link>
            </div>
          </div>
        </div>

        <div className="mt-8">
          <h2 className="text-xl font-semibold mb-4">Features</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="text-left">
              <h3 className="font-medium mb-2">🔄 Automated EDA</h3>
              <ul className="text-sm text-gray-600 space-y-1">
                <li>• Statistical summaries and data profiling</li>
                <li>• Missing value analysis</li>
                <li>• Distribution analysis and outlier detection</li>
                <li>• Correlation analysis</li>
                <li>• Interactive visualizations with Plotly</li>
              </ul>
            </div>
            <div className="text-left">
              <h3 className="font-medium mb-2">🤖 Machine Learning</h3>
              <ul className="text-sm text-gray-600 space-y-1">
                <li>• Auto-detect classification/regression tasks</li>
                <li>• Multiple algorithms (RF, LightGBM, etc.)</li>
                <li>• Hyperparameter optimization</li>
                <li>• Model comparison and selection</li>
                <li>• SHAP explainability</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home;
