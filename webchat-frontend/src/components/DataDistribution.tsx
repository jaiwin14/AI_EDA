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

interface DataDistributionProps {
  activeFileId: string | null;
  distributionData: any;
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

const DataDistribution: React.FC<DataDistributionProps> = ({
  activeFileId,
  distributionData,
  websocket,
  runAnalysis,
}) => {
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedColumn, setSelectedColumn] = useState<string>('');
  
  useEffect(() => {
    if (activeFileId && !distributionData && !isLoading) {
      handleRunAnalysis();
    }
  }, [activeFileId, distributionData]);
  
  useEffect(() => {
    if (distributionData && distributionData.columns && distributionData.columns.length > 0) {
      setSelectedColumn(distributionData.columns[0]);
    }
  }, [distributionData]);
  
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
      
      if (response.type === 'result' && response.function === 'data_distribution') {
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
            Data Distribution Analysis
          </Typography>
          <Alert severity="info">
            Please upload a file first to analyze data distribution.
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
            Data Distribution Analysis
          </Typography>
          <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
            <CircularProgress />
          </Box>
          <Typography variant="body2" align="center">
            Analyzing data distribution...
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
            Data Distribution Analysis
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
  
  if (!distributionData) {
    return (
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Data Distribution Analysis
          </Typography>
          <Alert severity="info" sx={{ mb: 2 }}>
            No distribution analysis available.
          </Alert>
          <Button
            variant="contained"
            startIcon={<RefreshIcon />}
            onClick={handleRunAnalysis}
          >
            Analyze Distribution
          </Button>
        </CardContent>
      </Card>
    );
  }
  
  // Get the selected column data
  const columnData = distributionData.distributions && distributionData.distributions[selectedColumn];
  
  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6">
            Data Distribution Analysis
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
        
        {distributionData.columns && distributionData.columns.length > 0 ? (
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
                  {distributionData.columns.map((column: string) => (
                    <MenuItem key={column} value={column}>
                      {column}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            
            {columnData ? (
              <>
                {/* Distribution Plot */}
                <Grid item xs={12}>
                  <Typography variant="subtitle1" gutterBottom>
                    Distribution for {selectedColumn}
                  </Typography>
                  <Paper variant="outlined" sx={{ p: 2, textAlign: 'center' }}>
                    {columnData.plot_image ? (
                      <StyledImage 
                        src={`data:image/png;base64,${columnData.plot_image}`} 
                        alt={`Distribution of ${selectedColumn}`} 
                      />
                    ) : (
                      <Alert severity="info">
                        No distribution plot available for this column.
                      </Alert>
                    )}
                  </Paper>
                </Grid>
                
                {/* Statistics */}
                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle1" gutterBottom>
                    Statistics
                  </Typography>
                  <TableContainer component={Paper} variant="outlined">
                    <Table size="small">
                      <TableBody>
                        {columnData.stats && Object.entries(columnData.stats).map(([stat, value]: [string, any]) => (
                          <TableRow key={stat}>
                            <TableCell component="th" scope="row">
                              {stat.charAt(0).toUpperCase() + stat.slice(1).replace('_', ' ')}
                            </TableCell>
                            <TableCell align="right">
                              {typeof value === 'number' ? value.toFixed(4) : value}
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </Grid>
                
                {/* Description */}
                <Grid item xs={12}>
                  <Typography variant="subtitle1" gutterBottom>
                    Description
                  </Typography>
                  <Paper variant="outlined" sx={{ p: 2 }}>
                    <Typography variant="body1">
                      {columnData.description || 'No description available.'}
                    </Typography>
                  </Paper>
                </Grid>
              </>
            ) : (
              <Grid item xs={12}>
                <Alert severity="info">
                  Select a column to view its distribution.
                </Alert>
              </Grid>
            )}
          </Grid>
        ) : (
          <Alert severity="info">
            No numeric columns found for distribution analysis.
          </Alert>
        )}
      </CardContent>
    </Card>
  );
};

export default DataDistribution;