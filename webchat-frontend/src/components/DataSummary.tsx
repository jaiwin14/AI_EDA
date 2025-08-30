import React, { useState, useEffect } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  Typography,
  Grid,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  CircularProgress,
  Alert,
  Divider,
} from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';

interface DataSummaryProps {
  activeFileId: string | null;
  summaryData: any;
  websocket: WebSocket | null;
  runAnalysis: () => void;
}

const DataSummary: React.FC<DataSummaryProps> = ({
  activeFileId,
  summaryData,
  websocket,
  runAnalysis,
}) => {
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  
  useEffect(() => {
    if (activeFileId && !summaryData && !isLoading) {
      handleRunAnalysis();
    }
  }, [activeFileId, summaryData]);
  
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
      
      if (response.type === 'result' && response.function === 'summary') {
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
            Data Summary
          </Typography>
          <Alert severity="info">
            Please upload a file first to view the data summary.
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
            Data Summary
          </Typography>
          <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
            <CircularProgress />
          </Box>
          <Typography variant="body2" align="center">
            Analyzing your data...
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
            Data Summary
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
  
  if (!summaryData) {
    return (
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Data Summary
          </Typography>
          <Alert severity="info" sx={{ mb: 2 }}>
            No summary data available.
          </Alert>
          <Button
            variant="contained"
            startIcon={<RefreshIcon />}
            onClick={handleRunAnalysis}
          >
            Generate Summary
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
            Data Summary
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
          {/* Dataset Statistics */}
          <Grid item xs={12} md={6}>
            <Typography variant="subtitle1" gutterBottom>
              Dataset Statistics
            </Typography>
            <TableContainer component={Paper} variant="outlined">
              <Table size="small">
                <TableBody>
                  <TableRow>
                    <TableCell component="th" scope="row">Number of variables</TableCell>
                    <TableCell align="right">{summaryData.num_variables}</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell component="th" scope="row">Number of observations</TableCell>
                    <TableCell align="right">{summaryData.num_observations}</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell component="th" scope="row">Missing cells</TableCell>
                    <TableCell align="right">{summaryData.missing_cells}</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell component="th" scope="row">Missing cells (%)</TableCell>
                    <TableCell align="right">{summaryData.missing_cells_percent}%</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell component="th" scope="row">Duplicate rows</TableCell>
                    <TableCell align="right">{summaryData.duplicate_rows}</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell component="th" scope="row">Duplicate rows (%)</TableCell>
                    <TableCell align="right">{summaryData.duplicate_rows_percent}%</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell component="th" scope="row">Total size in memory</TableCell>
                    <TableCell align="right">{summaryData.total_size}</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell component="th" scope="row">Average record size in memory</TableCell>
                    <TableCell align="right">{summaryData.average_record_size}</TableCell>
                  </TableRow>
                </TableBody>
              </Table>
            </TableContainer>
          </Grid>
          
          {/* Variable Types */}
          <Grid item xs={12} md={6}>
            <Typography variant="subtitle1" gutterBottom>
              Variable Types
            </Typography>
            <TableContainer component={Paper} variant="outlined">
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Type</TableCell>
                    <TableCell align="right">Count</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {Object.entries(summaryData.variable_types || {}).map(([type, count]: [string, any]) => (
                    <TableRow key={type}>
                      <TableCell component="th" scope="row">{type}</TableCell>
                      <TableCell align="right">{count}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Grid>
          
          {/* Column Information */}
          <Grid item xs={12}>
            <Typography variant="subtitle1" gutterBottom sx={{ mt: 2 }}>
              Column Information
            </Typography>
            <TableContainer component={Paper} variant="outlined">
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Column Name</TableCell>
                    <TableCell>Type</TableCell>
                    <TableCell align="right">Non-Null Count</TableCell>
                    <TableCell align="right">Missing Count</TableCell>
                    <TableCell align="right">Missing %</TableCell>
                    <TableCell align="right">Unique Values</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {summaryData.columns && Object.entries(summaryData.columns).map(([column, info]: [string, any]) => (
                    <TableRow key={column}>
                      <TableCell component="th" scope="row">{column}</TableCell>
                      <TableCell>{info.dtype}</TableCell>
                      <TableCell align="right">{info.non_null_count}</TableCell>
                      <TableCell align="right">{info.missing_count}</TableCell>
                      <TableCell align="right">{info.missing_percent}%</TableCell>
                      <TableCell align="right">{info.unique_count}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Grid>
          
          {/* Numeric Column Statistics */}
          {summaryData.numeric_stats && Object.keys(summaryData.numeric_stats).length > 0 && (
            <Grid item xs={12}>
              <Typography variant="subtitle1" gutterBottom sx={{ mt: 2 }}>
                Numeric Column Statistics
              </Typography>
              <TableContainer component={Paper} variant="outlined">
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Column</TableCell>
                      <TableCell align="right">Mean</TableCell>
                      <TableCell align="right">Std</TableCell>
                      <TableCell align="right">Min</TableCell>
                      <TableCell align="right">25%</TableCell>
                      <TableCell align="right">50% (Median)</TableCell>
                      <TableCell align="right">75%</TableCell>
                      <TableCell align="right">Max</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {Object.entries(summaryData.numeric_stats).map(([column, stats]: [string, any]) => (
                      <TableRow key={column}>
                        <TableCell component="th" scope="row">{column}</TableCell>
                        <TableCell align="right">{stats.mean?.toFixed(2)}</TableCell>
                        <TableCell align="right">{stats.std?.toFixed(2)}</TableCell>
                        <TableCell align="right">{stats.min?.toFixed(2)}</TableCell>
                        <TableCell align="right">{stats['25%']?.toFixed(2)}</TableCell>
                        <TableCell align="right">{stats['50%']?.toFixed(2)}</TableCell>
                        <TableCell align="right">{stats['75%']?.toFixed(2)}</TableCell>
                        <TableCell align="right">{stats.max?.toFixed(2)}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </Grid>
          )}
        </Grid>
      </CardContent>
    </Card>
  );
};

export default DataSummary;