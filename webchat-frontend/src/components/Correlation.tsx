import React, { useState, useEffect } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  Typography,
  Grid,
  CircularProgress,
  Alert,
  Paper,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from '@mui/material';
import { styled } from '@mui/material/styles';
import RefreshIcon from '@mui/icons-material/Refresh';

interface CorrelationProps {
  activeFileId: string | null;
  correlationData: any;
  websocket: WebSocket | null;
  runAnalysis: () => void;
}

const StyledImage = styled('img')(({ theme }) => ({
  maxWidth: '100%',
  height: 'auto',
  borderRadius: theme.shape.borderRadius,
  marginTop: theme.spacing(2),
  marginBottom: theme.spacing(2),
}));

const Correlation: React.FC<CorrelationProps> = ({
  activeFileId,
  correlationData,
  websocket,
  runAnalysis,
}) => {
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [correlationType, setCorrelationType] = useState<string>('pearson');
  
  useEffect(() => {
    if (activeFileId && !correlationData && !isLoading) {
      handleRunAnalysis();
    }
  }, [activeFileId, correlationData]);
  
  const handleRunAnalysis = () => {
    if (!activeFileId || !websocket) {
      setError('No active file or WebSocket connection');
      return;
    }
    
    setIsLoading(true);
    setError(null);
    
    // Set up WebSocket message handler
    const messageHandler = (event: MessageEvent) => {
      const response = JSON.parse(event.data);
      
      if (response.type === 'result' && response.function === 'correlation') {
        setIsLoading(false);
        websocket.removeEventListener('message', messageHandler);
      } else if (response.type === 'error') {
        setIsLoading(false);
        setError(response.text || 'An error occurred');
        websocket.removeEventListener('message', messageHandler);
      }
    };
    
    websocket.addEventListener('message', messageHandler);
    
    // Send message with correlation type
    const message = {
      action: 'run_analysis',
      function: 'correlation',
      file_id: activeFileId,
      params: {
        method: correlationType
      }
    };
    
    websocket.send(JSON.stringify(message));
  };
  
  const handleCorrelationTypeChange = (event: React.ChangeEvent<{ value: unknown }>) => {
    setCorrelationType(event.target.value as string);
    // Re-run analysis with new correlation type
    if (activeFileId && websocket) {
      handleRunAnalysis();
    }
  };
  
  if (!activeFileId) {
    return (
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Correlation Analysis
          </Typography>
          <Alert severity="info">
            Please upload a file first to analyze correlations.
          </Alert>
        </CardContent>
      </Card>
    );
  }
  
  if (isLoading) {
    return (
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Correlation Analysis
          </Typography>
          <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
            <CircularProgress />
          </Box>
          <Typography variant="body2" align="center">
            Analyzing correlations...
          </Typography>
        </CardContent>
      </Card>
    );
  }
  
  if (error) {
    return (
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Correlation Analysis
          </Typography>
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={handleRunAnalysis}
          >
            Try Again
          </Button>
        </CardContent>
      </Card>
    );
  }
  
  if (!correlationData) {
    return (
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Correlation Analysis
          </Typography>
          <Alert severity="info" sx={{ mb: 2 }}>
            No correlation analysis available.
          </Alert>
          <Button
            variant="contained"
            startIcon={<RefreshIcon />}
            onClick={handleRunAnalysis}
          >
            Analyze Correlations
          </Button>
        </CardContent>
      </Card>
    );
  }
  
  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6">
            Correlation Analysis
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <FormControl variant="outlined" size="small" sx={{ minWidth: 150, mr: 2 }}>
              <InputLabel>Correlation Type</InputLabel>
              <Select
                value={correlationType}
                onChange={handleCorrelationTypeChange}
                label="Correlation Type"
              >
                <MenuItem value="pearson">Pearson</MenuItem>
                <MenuItem value="spearman">Spearman</MenuItem>
                <MenuItem value="kendall">Kendall</MenuItem>
              </Select>
            </FormControl>
            <Button
              variant="outlined"
              size="small"
              startIcon={<RefreshIcon />}
              onClick={handleRunAnalysis}
            >
              Refresh
            </Button>
          </Box>
        </Box>
        
        <Grid container spacing={3}>
          {/* Correlation Matrix Visualization */}
          {correlationData.plot_image ? (
            <Grid item xs={12}>
              <Typography variant="subtitle1" gutterBottom>
                Correlation Matrix ({correlationType.charAt(0).toUpperCase() + correlationType.slice(1)})
              </Typography>
              <Paper variant="outlined" sx={{ p: 2, textAlign: 'center' }}>
                <StyledImage 
                  src={`data:image/png;base64,${correlationData.plot_image}`} 
                  alt="Correlation Matrix" 
                />
              </Paper>
            </Grid>
          ) : (
            <Grid item xs={12}>
              <Alert severity="info">
                No correlation matrix visualization available.
              </Alert>
            </Grid>
          )}
          
          {/* Top Correlations */}
          {correlationData.top_correlations && correlationData.top_correlations.length > 0 && (
            <Grid item xs={12}>
              <Typography variant="subtitle1" gutterBottom>
                Top Correlations
              </Typography>
              <Paper variant="outlined" sx={{ p: 2 }}>
                <ul>
                  {correlationData.top_correlations.map((corr: any, index: number) => (
                    <li key={index}>
                      <Typography variant="body1">
                        <strong>{corr.feature1}</strong> and <strong>{corr.feature2}</strong>: {corr.correlation.toFixed(4)}
                        {corr.interpretation && (
                          <span> - {corr.interpretation}</span>
                        )}
                      </Typography>
                    </li>
                  ))}
                </ul>
              </Paper>
            </Grid>
          )}
          
          {/* Insights */}
          {correlationData.insights && (
            <Grid item xs={12}>
              <Typography variant="subtitle1" gutterBottom>
                Insights
              </Typography>
              <Paper variant="outlined" sx={{ p: 2 }}>
                <Typography variant="body1">
                  {correlationData.insights}
                </Typography>
              </Paper>
            </Grid>
          )}
        </Grid>
      </CardContent>
    </Card>
  );
};

export default Correlation;