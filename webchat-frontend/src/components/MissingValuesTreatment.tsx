import React, { useState, useEffect } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  Typography,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  CircularProgress,
  Alert,
  Divider,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
} from '@mui/material';
import { styled } from '@mui/material/styles';
import DownloadIcon from '@mui/icons-material/Download';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import InfoIcon from '@mui/icons-material/Info';

interface MissingValuesTreatmentProps {
  activeFileId: string | null;
  treatmentData: any;
  onTreatmentApplied: (cleanedFileId: string) => void;
  websocket: WebSocket | null;
}

const StyledImage = styled('img')(({ theme }) => ({
  maxWidth: '100%',
  height: 'auto',
  borderRadius: theme.shape.borderRadius,
  marginTop: theme.spacing(2),
  marginBottom: theme.spacing(2),
}));

const TreatmentMethodOptions = [
  { value: 'auto', label: 'Auto (Recommended)' },
  { value: 'drop_rows', label: 'Drop Rows' },
  { value: 'drop_column', label: 'Drop Column' },
  { value: 'mean', label: 'Mean' },
  { value: 'median', label: 'Median' },
  { value: 'mode', label: 'Mode' },
  { value: 'constant', label: 'Constant Value' },
  { value: 'knn', label: 'KNN Imputation' },
  { value: 'none', label: 'No Treatment' },
];

const MissingValuesTreatment: React.FC<MissingValuesTreatmentProps> = ({
  activeFileId,
  treatmentData,
  onTreatmentApplied,
  websocket,
}) => {
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<boolean>(false);
  const [cleanedFileId, setCleanedFileId] = useState<string | null>(null);
  const [treatmentOptions, setTreatmentOptions] = useState<any>({});
  const [constantValues, setConstantValues] = useState<any>({});
  
  // Initialize treatment options from treatment data
  useEffect(() => {
    if (treatmentData?.treatment?.column_treatments) {
      const initialOptions: any = {};
      const initialConstants: any = {};
      
      Object.entries(treatmentData.treatment.column_treatments).forEach(([column, treatment]: [string, any]) => {
        initialOptions[column] = treatment.treatment_method || 'auto';
        initialConstants[column] = '';
      });
      
      setTreatmentOptions(initialOptions);
      setConstantValues(initialConstants);
    }
  }, [treatmentData]);
  
  const handleTreatmentMethodChange = (column: string, method: string) => {
    setTreatmentOptions(prev => ({
      ...prev,
      [column]: method,
    }));
  };
  
  const handleConstantValueChange = (column: string, value: string) => {
    setConstantValues(prev => ({
      ...prev,
      [column]: value,
    }));
  };
  
  const applyTreatment = () => {
    if (!activeFileId || !websocket) {
      setError('No active file or WebSocket connection');
      return;
    }
    
    setIsLoading(true);
    setError(null);
    setSuccess(false);
    
    // Prepare treatment configuration
    const treatmentConfig: any = {};
    
    Object.entries(treatmentOptions).forEach(([column, method]) => {
      treatmentConfig[column] = { method };
      
      // Add constant value if method is 'constant'
      if (method === 'constant' && constantValues[column]) {
        // Try to convert to number if possible
        const numValue = Number(constantValues[column]);
        treatmentConfig[column].fill_value = isNaN(numValue) ? constantValues[column] : numValue;
      }
    });
    
    // Send WebSocket message to apply treatment
    const message = {
      action: 'run_analysis',
      function: 'treat_missing_values',
      file_id: activeFileId,
      treatment_options: treatmentConfig,
    };
    
    // Set up WebSocket message handler
    const messageHandler = (event: MessageEvent) => {
      const response = JSON.parse(event.data);
      
      if (response.type === 'result' && response.function === 'treat_missing_values') {
        setIsLoading(false);
        
        if (response.success) {
          setSuccess(true);
          if (response.cleaned_file_id) {
            setCleanedFileId(response.cleaned_file_id);
            onTreatmentApplied(response.cleaned_file_id);
          }
        } else {
          setError(response.text || 'Failed to apply treatment');
        }
        
        // Remove this handler after processing
        websocket.removeEventListener('message', messageHandler);
      } else if (response.type === 'error') {
        setIsLoading(false);
        setError(response.text || 'An error occurred');
        websocket.removeEventListener('message', messageHandler);
      }
    };
    
    websocket.addEventListener('message', messageHandler);
    websocket.send(JSON.stringify(message));
  };
  
  const downloadCleanedFile = () => {
    if (!cleanedFileId) return;
    
    window.open(`http://localhost:8000/download/${cleanedFileId}?cleaned=true`, '_blank');
  };
  
  // If no treatment data is available
  if (!treatmentData || !treatmentData.treatment) {
    return (
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Missing Values Treatment
          </Typography>
          <Alert severity="info">
            Please run the missing values analysis first to get treatment recommendations.
          </Alert>
        </CardContent>
      </Card>
    );
  }
  
  // Get columns with missing values
  const columnsWithMissing = Object.keys(treatmentData.treatment.column_treatments || {});
  
  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Missing Values Treatment
        </Typography>
        
        {treatmentData.treatment.message && (
          <Alert severity="info" sx={{ mb: 2 }}>
            {treatmentData.treatment.message}
          </Alert>
        )}
        
        {columnsWithMissing.length === 0 ? (
          <Alert severity="success">
            No missing values found in the dataset. No treatment needed.
          </Alert>
        ) : (
          <>
            <Typography variant="subtitle1" gutterBottom>
              Configure Treatment Methods
            </Typography>
            
            <TableContainer component={Paper} sx={{ mb: 3 }}>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Column</TableCell>
                    <TableCell>Missing Count</TableCell>
                    <TableCell>Missing %</TableCell>
                    <TableCell>Treatment Method</TableCell>
                    <TableCell>Options</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {columnsWithMissing.map(column => {
                    const columnData = treatmentData.treatment.column_treatments[column];
                    return (
                      <TableRow key={column}>
                        <TableCell>{column}</TableCell>
                        <TableCell>{columnData.missing_percent.toFixed(2)}%</TableCell>
                        <TableCell>{columnData.missing_count}</TableCell>
                        <TableCell>
                          <FormControl fullWidth size="small">
                            <Select
                              value={treatmentOptions[column] || 'auto'}
                              onChange={(e) => handleTreatmentMethodChange(column, e.target.value)}
                            >
                              {TreatmentMethodOptions.map(option => (
                                <MenuItem key={option.value} value={option.value}>
                                  {option.label}
                                </MenuItem>
                              ))}
                            </Select>
                          </FormControl>
                        </TableCell>
                        <TableCell>
                          {treatmentOptions[column] === 'constant' && (
                            <TextField
                              size="small"
                              placeholder="Fill value"
                              value={constantValues[column] || ''}
                              onChange={(e) => handleConstantValueChange(column, e.target.value)}
                            />
                          )}
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </TableContainer>
            
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 2 }}>
              <Button
                variant="contained"
                color="primary"
                onClick={applyTreatment}
                disabled={isLoading}
                startIcon={isLoading ? <CircularProgress size={20} /> : null}
              >
                {isLoading ? 'Applying...' : 'Apply Treatment'}
              </Button>
              
              {cleanedFileId && (
                <Button
                  variant="outlined"
                  startIcon={<DownloadIcon />}
                  onClick={downloadCleanedFile}
                >
                  Download Cleaned Data
                </Button>
              )}
            </Box>
            
            {error && (
              <Alert severity="error" sx={{ mt: 2 }}>
                {error}
              </Alert>
            )}
            
            {success && (
              <Alert severity="success" sx={{ mt: 2 }}>
                Treatment applied successfully! The cleaned dataset is ready.
              </Alert>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
};

export default MissingValuesTreatment;