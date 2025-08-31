# AI EDA Setup Instructions

## Quick Setup

### 1. Environment Configuration

1. Copy the configuration template:

   ```bash
   cp config_template.env .env
   ```

2. Edit the `.env` file and add your Gemini API key:
   ```env
   GEMINI_API_KEY=your_actual_gemini_api_key_here
   ```

### 2. Get Your Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated API key
5. Paste it in your `.env` file

### 3. Install Dependencies

#### Backend (Python)

```bash
cd backend
pip install -r requirements.txt
```

#### Frontend (Node.js)

```bash
cd webchat-frontend
npm install
```

### 4. Test the Setup

Before starting the application, test if everything is working:

```bash
python test_backend.py
```

This will check:

- All required imports
- Database initialization
- Environment variables
- API key configuration

### 5. Start the Application

#### Option 1: Full Application (Backend + Frontend)

From the root directory:

```bash
python start.py
```

This will start both the backend server (port 8000) and frontend server (port 3000).

#### Option 2: Backend Only (if you have frontend issues)

From the root directory:

```bash
python start_backend_only.py
```

This will start only the backend server on http://localhost:8000
You can access the API documentation at http://localhost:8000/docs

## Features

- **Smart Data Analysis**: AI-powered insights using Gemini
- **Interactive Dashboard**: Modern, responsive UI
- **Step-by-step EDA**: Guided exploratory data analysis
- **Real-time Processing**: WebSocket-based communication
- **Data Cleaning**: Automated missing value and outlier treatment

## Troubleshooting

### "Gemini API key not found" Error

- Make sure you've created the `.env` file in the root directory
- Verify your API key is correct and active
- Check that the `.env` file is in the same folder as `start.py`
- Run `python test_backend.py` to diagnose import and configuration issues

### Connection Issues

- Ensure both backend (port 8000) and frontend (port 3000) are running
- Check your firewall settings
- Verify the WebSocket connection in the browser console
- If frontend fails to start, use `python start_backend_only.py` to run backend only

### Frontend Issues

- Make sure Node.js and npm are installed: https://nodejs.org/
- Check that npm is in your system PATH
- Try running `npm install` manually in the `webchat-frontend` directory

### File Upload Issues

- Supported formats: CSV, Excel (.xls, .xlsx)
- Maximum file size: 100MB
- Ensure your file has valid data structure

## API Endpoints

- `POST /upload` - Upload data files
- `GET /file/{file_id}/summary` - Get file summary
- `GET /file/{file_id}/missing` - Analyze missing values
- `POST /file/{file_id}/missing/treat` - Treat missing values
- `GET /file/{file_id}/univariate_analysis` - Univariate analysis
- `GET /file/{file_id}/bivariate_analysis` - Bivariate analysis
- `GET /file/{file_id}/outlier_detection_enhanced` - Outlier detection
- `GET /file/{file_id}/correlation` - Correlation analysis
- `WebSocket /ws` - Real-time analysis communication

## Development

### Backend Structure

```
backend/
├── server.py          # Main FastAPI server
├── database.py        # Database operations
├── eda/              # Analysis modules
├── routes/           # API routes
└── data/             # Data storage
```

### Frontend Structure

```
webchat-frontend/
├── src/
│   ├── app/          # Next.js pages
│   ├── components/   # React components
│   └── types/        # TypeScript types
└── public/           # Static assets
```

## Support

For issues and questions:

1. Check the troubleshooting section above
2. Review the console logs for error messages
3. Ensure all dependencies are properly installed
4. Verify your API key is valid and has sufficient quota
