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
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from '@mui/material';
import { styled } from '@mui/material/styles';
import RefreshIcon from '@mui/icons-material/Refresh';

interface MissingValuesProps {
  activeFileId: string | null;
  missingData: any;
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

const MissingValues: React.FC<MissingValuesProps> = ({
  activeFileId,
  missingData,
  websocket,
  runAnalysis,
}) => {
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  
  useEffect(() => {
    if (activeFileId && !missingData && !isLoading) {
      handleRunAnalysis();
    }
  }, [activeFileId, missingData]);
  
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
      
      if (response.type === 'result' && response.function === 'missing_values') {
        setIsLoading(false);
        websocket.removeEventListener('message', messageHandler);
      } else if (response.type === 'error') {
        setIsLoading(false);
        setError(response.text || 'An error occurred');
        websocket.removeEventListener('message', messageHandler);
      }
    };
    
    websocket.addEventListener('message', messageHandler);
    runAnalysis();
  };
  
  if (!activeFileId) {
    return (
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Missing Values Analysis
          </Typography>
          <Alert severity="info">
            Please upload a file first to analyze missing values.
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
            Missing Values Analysis
          </Typography>
          <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
            <CircularProgress />
          </Box>
          <Typography variant="body2" align="center">
            Analyzing missing values...
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
            Missing Values Analysis
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
  
  if (!missingData) {
    return (
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Missing Values Analysis
          </Typography>
          <Alert severity="info" sx={{ mb: 2 }}>
            No missing values analysis available.
          </Alert>
          <Button
            variant="contained"
            startIcon={<RefreshIcon />}
            onClick={handleRunAnalysis}
          >
            Analyze Missing Values
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
            Missing Values Analysis
          </Typography>
          <Button
            variant="outlined"
            size="small"
            startIcon={<RefreshIcon />}
            onClick={handleRunAnalysis}
          >
            Refresh
          </Button>
        </Box>
        
        <Grid container spacing={3}>
          {/* Summary */}
          <Grid item xs={12}>
            <Typography variant="subtitle1" gutterBottom>
              Summary
            </Typography>
            <Paper variant="outlined" sx={{ p: 2 }}>
              <Typography variant="body1">
                {missingData.summary_text || 'No summary available.'}
              </Typography>
            </Paper>
          </Grid>
          
          {/* Missing Values Visualization */}
          {missingData.plot_image && (
            <Grid item xs={12}>
              <Typography variant="subtitle1" gutterBottom>
                Missing Values Visualization
              </Typography>
              <Paper variant="outlined" sx={{ p: 2, textAlign: 'center' }}>
                <StyledImage src={`data:image/png;base64,${missingData.plot_image}`} alt="Missing Values Plot" />
              </Paper>
            </Grid>
          )}
          
          {/* Missing Values Table */}
          <Grid item xs={12}>
            <Typography variant="subtitle1" gutterBottom>
              Missing Values by Column
            </Typography>
            <TableContainer component={Paper} variant="outlined">
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Column</TableCell>
                    <TableCell align="right">Missing Count</TableCell>
                    <TableCell align="right">Missing %</TableCell>
                    <TableCell>Data Type</TableCell>
                    <TableCell>Recommended Treatment</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {missingData.missing_columns && missingData.missing_columns.map((column: any) => (
                    <TableRow key={column.column_name}>
                      <TableCell component="th" scope="row">{column.column_name}</TableCell>
                      <TableCell align="right">{column.missing_count}</TableCell>
                      <TableCell align="right">{column.missing_percent.toFixed(2)}%</TableCell>
                      <TableCell>{column.dtype}</TableCell>
                      <TableCell>{column.recommended_treatment || 'N/A'}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Grid>
          
          {/* Treatment Recommendations */}
          {missingData.treatment && (
            <Grid item xs={12}>
              <Typography variant="subtitle1" gutterBottom>
                Treatment Recommendations
              </Typography>
              <Paper variant="outlined" sx={{ p: 2 }}>
                <Typography variant="body1">
                  {missingData.treatment.message || 'No treatment recommendations available.'}
                </Typography>
              </Paper>
            </Grid>
          )}
        </Grid>
      </CardContent>
    </Card>
  );
};

export default MissingValues;