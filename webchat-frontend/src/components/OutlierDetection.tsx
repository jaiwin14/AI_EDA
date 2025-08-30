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
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from '@mui/material';
import { styled } from '@mui/material/styles';
import RefreshIcon from '@mui/icons-material/Refresh';

interface OutlierDetectionProps {
  activeFileId: string | null;
  outlierData: any;
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

const OutlierDetection: React.FC<OutlierDetectionProps> = ({
  activeFileId,
  outlierData,
  websocket,
  runAnalysis,
}) => {
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedColumn, setSelectedColumn] = useState<string>('');
  
  useEffect(() => {
    if (activeFileId && !outlierData && !isLoading) {
      handleRunAnalysis();
    }
  }, [activeFileId, outlierData]);
  
  useEffect(() => {
    if (outlierData && outlierData.columns && outlierData.columns.length > 0) {
      setSelectedColumn(outlierData.columns[0]);
    }
  }, [outlierData]);
  
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
      
      if (response.type === 'result' && response.function === 'outlier_detection') {
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
  
  const handleColumnChange = (event: React.ChangeEvent<{ value: unknown }>) => {
    setSelectedColumn(event.target.value as string);
  };
  
  if (!activeFileId) {
    return (
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Outlier Detection
          </Typography>
          <Alert severity="info">
            Please upload a file first to detect outliers.
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
            Outlier Detection
          </Typography>
          <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
            <CircularProgress />
          </Box>
          <Typography variant="body2" align="center">
            Detecting outliers...
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
            Outlier Detection
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
  
  if (!outlierData) {
    return (
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Outlier Detection
          </Typography>
          <Alert severity="info" sx={{ mb: 2 }}>
            No outlier detection results available.
          </Alert>
          <Button
            variant="contained"
            startIcon={<RefreshIcon />}
            onClick={handleRunAnalysis}
          >
            Detect Outliers
          </Button>
        </CardContent>
      </Card>
    );
  }
  
  // Get the selected column data
  const columnData = outlierData.outliers && outlierData.outliers[selectedColumn];
  
  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6">
            Outlier Detection
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
        
        {outlierData.columns && outlierData.columns.length > 0 ? (
          <Grid container spacing={3}>
            {/* Column Selector */}
            <Grid item xs={12}>
              <FormControl fullWidth variant="outlined" size="small">
                <InputLabel>Select Column</InputLabel>
                <Select
                  value={selectedColumn}
                  onChange={handleColumnChange}
                  label="Select Column"
                >
                  {outlierData.columns.map((column: string) => (
                    <MenuItem key={column} value={column}>
                      {column}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            
            {columnData ? (
              <>
                {/* Summary */}
                <Grid item xs={12}>
                  <Typography variant="subtitle1" gutterBottom>
                    Outlier Summary for {selectedColumn}
                  </Typography>
                  <Paper variant="outlined" sx={{ p: 2 }}>
                    <Typography variant="body1">
                      {columnData.summary_text || 'No summary available.'}
                    </Typography>
                  </Paper>
                </Grid>
                
                {/* Outlier Plot */}
                <Grid item xs={12}>
                  <Typography variant="subtitle1" gutterBottom>
                    Outlier Visualization
                  </Typography>
                  <Paper variant="outlined" sx={{ p: 2, textAlign: 'center' }}>
                    {columnData.plot_image ? (
                      <StyledImage 
                        src={`data:image/png;base64,${columnData.plot_image}`} 
                        alt={`Outliers in ${selectedColumn}`} 
                      />
                    ) : (
                      <Alert severity="info">
                        No outlier plot available for this column.
                      </Alert>
                    )}
                  </Paper>
                </Grid>
                
                {/* Outlier Statistics */}
                <Grid item xs={12}>
                  <Typography variant="subtitle1" gutterBottom>
                    Outlier Statistics
                  </Typography>
                  <TableContainer component={Paper} variant="outlined">
                    <Table size="small">
                      <TableBody>
                        {columnData.stats && (
                          <>
                            <TableRow>
                              <TableCell component="th" scope="row">Outlier Count</TableCell>
                              <TableCell align="right">{columnData.stats.outlier_count}</TableCell>
                            </TableRow>
                            <TableRow>
                              <TableCell component="th" scope="row">Outlier Percentage</TableCell>
                              <TableCell align="right">{columnData.stats.outlier_percentage.toFixed(2)}%</TableCell>
                            </TableRow>
                            <TableRow>
                              <TableCell component="th" scope="row">Lower Bound</TableCell>
                              <TableCell align="right">{columnData.stats.lower_bound.toFixed(4)}</TableCell>
                            </TableRow>
                            <TableRow>
                              <TableCell component="th" scope="row">Upper Bound</TableCell>
                              <TableCell align="right">{columnData.stats.upper_bound.toFixed(4)}</TableCell>
                            </TableRow>
                            <TableRow>
                              <TableCell component="th" scope="row">Min Outlier Value</TableCell>
                              <TableCell align="right">{columnData.stats.min_outlier?.toFixed(4) || 'N/A'}</TableCell>
                            </TableRow>
                            <TableRow>
                              <TableCell component="th" scope="row">Max Outlier Value</TableCell>
                              <TableCell align="right">{columnData.stats.max_outlier?.toFixed(4) || 'N/A'}</TableCell>
                            </TableRow>
                          </>
                        )}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </Grid>
              </>
            ) : (
              <Grid item xs={12}>
                <Alert severity="info">
                  Select a column to view its outlier analysis.
                </Alert>
              </Grid>
            )}
          </Grid>
        ) : (
          <Alert severity="info">
            No numeric columns found for outlier detection.
          </Alert>
        )}
      </CardContent>
    </Card>
  );
};

export default OutlierDetection;