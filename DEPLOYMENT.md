# 🚀 Deployment Guide

This guide covers deploying the AI EDA Platform to free-tier services.

## Prerequisites

- GitHub account
- Gemini API key from Google AI Studio
- Supabase account (optional, for production storage)

## 1. Backend Deployment (Render.com)

### Step 1: Prepare Repository
1. Push your code to GitHub
2. Ensure `render.yaml` is in the backend directory

### Step 2: Deploy to Render
1. Go to [Render.com](https://render.com) and sign up
2. Connect your GitHub account
3. Create a new "Web Service"
4. Select your repository
5. Use these settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Environment**: Python 3.11

### Step 3: Set Environment Variables
In Render dashboard, add these environment variables:
```
GEMINI_API_KEY=your_actual_gemini_api_key
SUPABASE_URL=your_supabase_url (optional)
SUPABASE_KEY=your_supabase_key (optional)
ENVIRONMENT=production
```

### Step 4: Deploy
- Click "Create Web Service"
- Wait for deployment (5-10 minutes)
- Note your backend URL: `https://your-app-name.onrender.com`

## 2. Frontend Deployment (Netlify)

### Step 1: Prepare Frontend
1. Update `netlify.toml` with your backend URL
2. Ensure build command is set to `npm run build`

### Step 2: Deploy to Netlify
1. Go to [Netlify](https://netlify.com) and sign up
2. Connect your GitHub account
3. Create new site from Git
4. Select your repository
5. Use these settings:
   - **Build Command**: `npm run build`
   - **Publish Directory**: `dist`
   - **Base Directory**: `frontend`

### Step 3: Set Environment Variables
In Netlify dashboard, add:
```
VITE_API_URL=https://your-backend-url.onrender.com
```

### Step 4: Deploy
- Click "Deploy Site"
- Wait for build completion
- Your frontend will be available at: `https://your-site-name.netlify.app`

## 3. Database Setup (Supabase - Optional)

### Step 1: Create Supabase Project
1. Go to [Supabase](https://supabase.com)
2. Create new project
3. Note your project URL and anon key

### Step 2: Create Tables
Run these SQL commands in Supabase SQL editor:

```sql
-- Datasets table
CREATE TABLE datasets (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    dataset_id VARCHAR UNIQUE NOT NULL,
    filename VARCHAR NOT NULL,
    upload_time TIMESTAMP DEFAULT NOW(),
    file_size BIGINT,
    status VARCHAR DEFAULT 'uploaded',
    metadata JSONB
);

-- EDA results table
CREATE TABLE eda_results (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    dataset_id VARCHAR REFERENCES datasets(dataset_id),
    analyses JSONB,
    visualizations JSONB,
    summary TEXT,
    narrative TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Models table
CREATE TABLE models (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    model_id VARCHAR UNIQUE NOT NULL,
    dataset_id VARCHAR,
    model_name VARCHAR NOT NULL,
    model_type VARCHAR,
    target_column VARCHAR,
    task_type VARCHAR,
    status VARCHAR DEFAULT 'training',
    metrics JSONB,
    training_time FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Predictions table
CREATE TABLE predictions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    model_id VARCHAR,
    input_data JSONB,
    prediction JSONB,
    confidence FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

## 4. CI/CD Setup (GitHub Actions)

The project includes GitHub Actions workflow in `.github/workflows/deploy.yml`.

### Automatic Deployment
- Push to `main` branch triggers deployment
- Tests run automatically
- Both frontend and backend deploy on successful tests

### Manual Deployment
You can also trigger deployments manually from GitHub Actions tab.

## 5. Monitoring Setup (Sentry - Optional)

### Step 1: Create Sentry Account
1. Go to [Sentry.io](https://sentry.io)
2. Create new project for Python (backend) and React (frontend)

### Step 2: Add Sentry to Backend
Add to your Render environment variables:
```
SENTRY_DSN=your_backend_sentry_dsn
```

### Step 3: Add Sentry to Frontend
Add to your Netlify environment variables:
```
VITE_SENTRY_DSN=your_frontend_sentry_dsn
```

## 6. Custom Domain (Optional)

### Netlify Custom Domain
1. In Netlify dashboard, go to Domain settings
2. Add custom domain
3. Follow DNS configuration instructions

### Render Custom Domain
1. In Render dashboard, go to Settings
2. Add custom domain
3. Configure DNS records

## 7. SSL Certificates

Both Render and Netlify provide free SSL certificates automatically.

## 8. Environment Variables Summary

### Backend (Render)
```
GEMINI_API_KEY=your_gemini_api_key
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
ENVIRONMENT=production
SENTRY_DSN=your_sentry_dsn (optional)
```

### Frontend (Netlify)
```
VITE_API_URL=https://your-backend.onrender.com
VITE_SENTRY_DSN=your_sentry_dsn (optional)
```

## 9. Troubleshooting

### Backend Issues
- Check Render logs for errors
- Verify environment variables are set
- Ensure Gemini API key is valid

### Frontend Issues
- Check Netlify build logs
- Verify VITE_API_URL is correct
- Check browser console for errors

### Database Issues
- Verify Supabase connection
- Check table permissions
- Review SQL queries in logs

## 10. Scaling Considerations

### Free Tier Limits
- **Render**: 750 hours/month, sleeps after 15min inactivity
- **Netlify**: 125k requests/month, 100GB bandwidth
- **Supabase**: 500MB database, 2GB bandwidth

### Upgrading
- Monitor usage in dashboards
- Upgrade to paid plans as needed
- Consider CDN for static assets

## 11. Backup Strategy

### Code Backup
- Code is backed up in GitHub
- Use multiple branches for different environments

### Data Backup
- Supabase provides automatic backups
- Export important data regularly
- Consider additional backup solutions for production

## 12. Security Best Practices

- Never commit API keys to Git
- Use environment variables for all secrets
- Enable 2FA on all accounts
- Regularly rotate API keys
- Monitor access logs

## Support

For deployment issues:
1. Check service status pages
2. Review documentation
3. Contact support if needed
4. Use community forums

Remember to test your deployment thoroughly before going live!
