AI_EDA

# 🤖 AI EDA Platform

A production-ready end-to-end machine learning web application for automated exploratory data analysis, feature engineering, multi-model training, and intelligent predictions with SHAP explainability.

## ✨ Features

### 🔍 Automated EDA
- **Statistical Analysis**: Comprehensive data profiling and quality assessment
- **Missing Values**: Intelligent analysis and treatment recommendations
- **Distribution Analysis**: Skewness, kurtosis, and normality testing
- **Correlation Analysis**: Feature relationships and multicollinearity detection
- **Outlier Detection**: Multiple methods with severity assessment
- **Interactive Visualizations**: Plotly-powered charts and graphs

### 🤖 Machine Learning Pipeline
- **Auto Task Detection**: Automatic classification/regression identification
- **Multi-Model Training**: Random Forest, LightGBM, Logistic/Linear Regression
- **Hyperparameter Optimization**: Optuna-powered automated tuning
- **Cross-Validation**: Robust model evaluation and selection
- **Feature Engineering**: Automated preprocessing and encoding
- **Model Comparison**: Side-by-side performance metrics

### 🎯 Model Explainability
- **SHAP Integration**: Local and global explanations
- **Feature Importance**: Understand model decision-making
- **Prediction Confidence**: Uncertainty quantification
- **Interactive Plots**: Waterfall and summary plots

### 🧠 AI-Powered Narratives
- **Gemini API Integration**: Automated report generation
- **EDA Summaries**: Natural language insights
- **Model Explanations**: Plain English model behavior
- **Recommendations**: Data preprocessing and modeling advice

### 🌐 Production-Ready Web App
- **React Frontend**: Modern, responsive UI with TypeScript
- **FastAPI Backend**: High-performance async API
- **Real-time Updates**: Live training progress and status
- **File Upload**: Drag-and-drop CSV/XLS/XLSX support
- **Interactive Dashboards**: Comprehensive data exploration

## 🛠 Tech Stack

### Backend
- **FastAPI** + Uvicorn - High-performance async web framework
- **Python 3.11+** - Modern Python with type hints
- **scikit-learn** - Core ML algorithms and preprocessing
- **LightGBM** - Gradient boosting for advanced models
- **SHAP** - Model explainability and interpretability
- **Optuna** - Hyperparameter optimization
- **Plotly** - Interactive visualization generation
- **Supabase** - PostgreSQL database and storage

### Frontend
- **React 18** + TypeScript - Modern component-based UI
- **Vite** - Fast build tool and dev server
- **React Query** - Server state management
- **React Router** - Client-side routing
- **Plotly.js** - Interactive data visualizations
- **React Dropzone** - File upload interface
- **Axios** - HTTP client for API calls

### Deployment & DevOps
- **Render.com** - Backend hosting (free tier)
- **Netlify** - Frontend hosting and CDN (free tier)
- **GitHub Actions** - CI/CD pipeline
- **Sentry** - Error monitoring and performance tracking

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- Git

### Backend Setup

```bash
# Clone the repository
git clone <repository-url>
cd AI_EDA-main

# Set up backend
cd backend
pip install -r requirements.txt

# Create environment file
cp .env.example .env
# Edit .env with your API keys

# Start the backend server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup

```bash
# Set up frontend (in new terminal)
cd frontend
npm install

# Create environment file
cp .env.example .env
# Edit .env with backend URL

# Start the frontend development server
npm run dev
```

### Access the Application
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 🔧 Environment Variables

### Backend (.env)
```bash
# Required
GEMINI_API_KEY=your_gemini_api_key_here
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_anon_key

# Optional
ENVIRONMENT=development
UPLOAD_DIR=./uploads
MODELS_DIR=./models
MAX_FILE_SIZE=104857600  # 100MB
CORS_ORIGINS=["http://localhost:3000"]
```

### Frontend (.env)
```bash
VITE_API_URL=http://localhost:8000
```

## 📊 Usage Guide

### 1. Upload Dataset
- Navigate to **Upload Data** page
- Drag & drop or select CSV/XLS/XLSX files (max 100MB)
- Supported formats: CSV, Excel (.xls, .xlsx)
- First row should contain column headers

### 2. Explore Data (EDA)
- Automatic data profiling and quality assessment
- Interactive visualizations for distributions, correlations
- Missing value analysis and treatment recommendations
- Outlier detection and statistical summaries
- AI-generated narrative insights

### 3. Train Models
- Select target column for prediction
- Choose task type (auto-detect, classification, regression)
- Automated training of multiple algorithms
- Real-time progress tracking
- Model comparison and selection

### 4. Make Predictions
- Select trained model
- Input feature values for single predictions
- View confidence scores and explanations
- Batch prediction support (coming soon)

### 5. Model Explanations
- SHAP-powered local and global explanations
- Feature importance analysis
- Interactive explanation plots
- Model behavior insights

## 🚀 Deployment

### Render.com (Backend)
1. Connect your GitHub repository to Render
2. Create a new Web Service
3. Use the provided `render.yaml` configuration
4. Set environment variables in Render dashboard
5. Deploy automatically on git push

### Netlify (Frontend)
1. Connect your GitHub repository to Netlify
2. Set build command: `npm run build`
3. Set publish directory: `dist`
4. Use the provided `netlify.toml` configuration
5. Deploy automatically on git push

### Environment Variables Setup
- **Gemini API**: Get API key from Google AI Studio
- **Supabase**: Create project and get URL + anon key
- **Sentry** (optional): Create project for error monitoring

## 📡 API Endpoints

### Upload & Data Management
- `POST /api/upload/` - Upload dataset file
- `GET /api/upload/datasets` - List uploaded datasets
- `GET /api/upload/datasets/{dataset_id}` - Get dataset info
- `DELETE /api/upload/datasets/{dataset_id}` - Delete dataset

### Exploratory Data Analysis
- `POST /api/eda/{dataset_id}/run` - Start EDA analysis
- `GET /api/eda/{dataset_id}/results` - Get EDA results
- `GET /api/eda/{dataset_id}/visualizations` - Get visualizations
- `GET /api/eda/{dataset_id}/summary` - Get analysis summary
- `GET /api/eda/{dataset_id}/report` - Get HTML report

### Model Training & Management
- `POST /api/models/{dataset_id}/train` - Start model training
- `GET /api/models/{dataset_id}` - List trained models
- `GET /api/models/{dataset_id}/status` - Get training status
- `GET /api/models/{dataset_id}/compare` - Compare model performance
- `DELETE /api/models/{model_id}` - Delete trained model

### Predictions
- `POST /api/predictions/{model_id}/predict` - Single prediction
- `POST /api/predictions/{model_id}/batch` - Batch predictions
- `GET /api/predictions/{model_id}/schema` - Get input schema
- `GET /api/predictions/history` - Prediction history

### Model Explanations
- `POST /api/explanations/{model_id}/explain` - Single explanation
- `POST /api/explanations/{model_id}/batch` - Batch explanations
- `GET /api/explanations/{model_id}/global` - Global explanations
- `GET /api/explanations/{model_id}/plots` - Explanation plots

### Health & Status
- `GET /health` - Health check endpoint
- `GET /` - API root with welcome message

## 🧪 Development

### Running Tests
```bash
# Backend tests
cd backend
python -m pytest tests/ -v

# Frontend tests
cd frontend
npm test
```

### Code Quality
```bash
# Backend formatting
cd backend
black app/
isort app/

# Frontend linting
cd frontend
npm run lint
npm run type-check
```

### Adding New Features
1. Backend: Add new endpoints in `app/api/`
2. Frontend: Add new components in `src/components/`
3. Update API client and types
4. Add tests for new functionality
5. Update documentation

## 🔍 Troubleshooting

### Common Issues

**Backend won't start:**
- Check Python version (3.11+ required)
- Verify all dependencies installed: `pip install -r requirements.txt`
- Check environment variables in `.env` file
- Ensure ports 8000 is available

**Frontend won't start:**
- Check Node.js version (18+ required)
- Clear node_modules: `rm -rf node_modules && npm install`
- Check environment variables in `.env` file
- Ensure port 3000 is available

**Upload fails:**
- Check file size (max 100MB)
- Verify file format (CSV, XLS, XLSX only)
- Ensure first row contains headers
- Check backend logs for detailed errors

**Model training fails:**
- Verify target column exists and has valid data
- Check for sufficient data (minimum 100 rows recommended)
- Ensure numeric columns for regression tasks
- Check backend logs for detailed errors

### Getting Help
- Check the API documentation at `/docs`
- Review backend logs for detailed error messages
- Ensure all environment variables are properly set
- Verify network connectivity between frontend and backend

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Commit your changes: `git commit -am 'Add feature'`
5. Push to the branch: `git push origin feature-name`
6. Submit a pull request

## 🙏 Acknowledgments

- **Streamlit Community** - Inspiration for the original EDA app
- **FastAPI** - Amazing web framework for Python
- **React Team** - Excellent frontend library
- **Plotly** - Beautiful interactive visualizations
- **SHAP** - Model explainability made simple
- **Render & Netlify** - Free hosting for developers
