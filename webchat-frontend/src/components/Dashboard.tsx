import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Container,
  Grid,
  Paper,
  Typography,
  Button,
  AppBar,
  Toolbar,
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
  CircularProgress,
  Alert,
  Stepper,
  Step,
  StepLabel,
  Card,
  CardContent,
  IconButton,
} from '@mui/material';
import { styled } from '@mui/material/styles';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import TableChartIcon from '@mui/icons-material/TableChart';
import BarChartIcon from '@mui/icons-material/BarChart';
import BubbleChartIcon from '@mui/icons-material/BubbleChart';
import CleaningServicesIcon from '@mui/icons-material/CleaningServices';
import WarningIcon from '@mui/icons-material/Warning';
import MenuIcon from '@mui/icons-material/Menu';
import ChevronLeftIcon from '@mui/icons-material/ChevronLeft';
import ChevronRightIcon from '@mui/icons-material/ChevronRight';
import DownloadIcon from '@mui/icons-material/Download';
import FileUpload from './FileUpload';
import DataSummary from './DataSummary';
import MissingValues from './MissingValues';
import MissingValuesTreatment from './MissingValuesTreatment';
import DataDistribution from './DataDistribution';
import OutlierDetection from './OutlierDetection';
import Correlation from './Correlation';

const drawerWidth = 240;

const Main = styled('main', { shouldForwardProp: (prop) => prop !== 'open' })(
  ({ theme, open }: { theme: any; open: boolean }) => ({
    flexGrow: 1,
    padding: theme.spacing(3),
    transition: theme.transitions.create('margin', {
      easing: theme.transitions.easing.sharp,
      duration: theme.transitions.duration.leavingScreen,
    }),
    marginLeft: `-${drawerWidth}px`,
    ...(open && {
      transition: theme.transitions.create('margin', {
        easing: theme.transitions.easing.easeOut,
        duration: theme.transitions.duration.enteringScreen,
      }),
      marginLeft: 0,
    }),
  }),
);

const DrawerHeader = styled('div')(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  padding: theme.spacing(0, 1),
  ...theme.mixins.toolbar,
  justifyContent: 'flex-end',
}));

const StyledAppBar = styled(AppBar, {
  shouldForwardProp: (prop) => prop !== 'open',
})(
  ({ theme, open }: { theme: any; open: boolean }) => ({
    transition: theme.transitions.create(['margin', 'width'], {
      easing: theme.transitions.easing.sharp,
      duration: theme.transitions.duration.leavingScreen,
    }),
    ...(open && {
      width: `calc(100% - ${drawerWidth}px)`,
      marginLeft: `${drawerWidth}px`,
      transition: theme.transitions.create(['margin', 'width'], {
        easing: theme.transitions.easing.easeOut,
        duration: theme.transitions.duration.enteringScreen,
      }),
    }),
  }),
);

const steps = [
  'Upload Data',
  'Data Summary',
  'Missing Values',
  'Treat Missing Values',
  'Data Distribution',
  'Outlier Detection',
  'Correlation Analysis',
];

const Dashboard: React.FC = () => {
  const [activeStep, setActiveStep] = useState<number>(0);
  const [drawerOpen, setDrawerOpen] = useState<boolean>(true);
  const [activeFileId, setActiveFileId] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string>('');
  const [websocket, setWebsocket] = useState<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [analysisResults, setAnalysisResults] = useState<{
    summary?: any;
    missing_values?: any;
    data_distribution?: any;
    outlier_detection?: any;
    correlation?: any;
    treatment?: any;
  }>({});
  const [cleanedFileId, setCleanedFileId] = useState<string | null>(null);
  
  // WebSocket connection
  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws');
    
    ws.onopen = () => {
      console.log('WebSocket connected');
      setIsConnected(true);
      setWebsocket(ws);
    };
    
    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setIsConnected(false);
      setWebsocket(null);
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setError('Failed to connect to the server. Please try again.');
    };
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log('WebSocket message:', data);
        
        if (data.type === 'result') {
          // Handle analysis results
          if (data.function && data.success) {
            setAnalysisResults(prev => ({
              ...prev,
              [data.function]: data.result,
            }));
            
            // If this is a treatment result, store the cleaned file ID
            if (data.function === 'treat_missing_values' && data.cleaned_file_id) {
              setCleanedFileId(data.cleaned_file_id);
            }
          } else if (data.type === 'error') {
            setError(data.text || 'An error occurred during analysis');
          }
        }
      } catch (err) {
        console.error('Error parsing WebSocket message:', err);
      }
    };
    
    return () => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    };
  }, []);
  
  const handleFileUpload = (file: File, fileId: string) => {
    setActiveFileId(fileId);
    setFileName(file.name);
    setActiveStep(1); // Move to Data Summary step after upload
    
    // Reset analysis results when a new file is uploaded
    setAnalysisResults({});
    setCleanedFileId(null);
  };
  
  const handleDrawerOpen = () => {
    setDrawerOpen(true);
  };
  
  const handleDrawerClose = () => {
    setDrawerOpen(false);
  };
  
  const handleStepChange = (step: number) => {
    setActiveStep(step);
  };
  
  const handleNext = () => {
    setActiveStep((prevStep) => Math.min(prevStep + 1, steps.length - 1));
  };
  
  const handleBack = () => {
    setActiveStep((prevStep) => Math.max(prevStep - 1, 0));
  };
  
  const runAnalysis = (functionName: string) => {
    if (!activeFileId || !websocket || websocket.readyState !== WebSocket.OPEN) {
      setError('No active file or WebSocket connection');
      return;
    }
    
    // Use cleaned file ID if available and appropriate
    const fileId = (cleanedFileId && activeStep > 3) ? cleanedFileId : activeFileId;
    
    const message = {
      action: 'run_analysis',
      function: functionName,
      file_id: fileId,
    };
    
    websocket.send(JSON.stringify(message));
  };
  
  const handleTreatmentApplied = (newCleanedFileId: string) => {
    setCleanedFileId(newCleanedFileId);
  };
  
  const downloadFile = (isCleanedFile: boolean = false) => {
    const fileId = isCleanedFile ? cleanedFileId : activeFileId;
    if (!fileId) {
      setError('No file available for download');
      return;
    }
    
    // Ensure fileId is properly encoded in the URL
    const encodedFileId = encodeURIComponent(fileId);
    const url = `http://localhost:8000/download/${encodedFileId}${isCleanedFile ? '?cleaned=true' : ''}`;
    console.log(`Downloading file from: ${url}`);
    window.open(url, '_blank');
  };
  
  // Render the current step content
  const renderStepContent = () => {
    switch (activeStep) {
      case 0: // Upload Data
        return <FileUpload onFileUploaded={handleFileUpload} />;
      
      case 1: // Data Summary
        return (
          <DataSummary 
            activeFileId={activeFileId} 
            summaryData={analysisResults.summary} 
            websocket={websocket} 
            runAnalysis={() => runAnalysis('summary')} 
          />
        );
      
      case 2: // Missing Values
        return (
          <MissingValues 
            activeFileId={activeFileId} 
            missingData={analysisResults.missing_values} 
            websocket={websocket} 
            runAnalysis={() => runAnalysis('missing_values')} 
          />
        );
      
      case 3: // Treat Missing Values
        return (
          <MissingValuesTreatment 
            activeFileId={activeFileId} 
            treatmentData={analysisResults.missing_values} 
            onTreatmentApplied={handleTreatmentApplied} 
            websocket={websocket} 
          />
        );
      
      case 4: // Data Distribution
        return (
          <DataDistribution 
            activeFileId={cleanedFileId || activeFileId} 
            distributionData={analysisResults.data_distribution} 
            websocket={websocket} 
            runAnalysis={() => runAnalysis('data_distribution')} 
          />
        );
      
      case 5: // Outlier Detection
        return (
          <OutlierDetection 
            activeFileId={cleanedFileId || activeFileId} 
            outlierData={analysisResults.outlier_detection} 
            websocket={websocket} 
            runAnalysis={() => runAnalysis('outlier_detection')} 
          />
        );
      
      case 6: // Correlation Analysis
        return (
          <Correlation 
            activeFileId={cleanedFileId || activeFileId} 
            correlationData={analysisResults.correlation} 
            websocket={websocket} 
            runAnalysis={() => runAnalysis('correlation')} 
          />
        );
      
      default:
        return <Typography>Unknown step</Typography>;
    }
  };
  
  return (
    <Box sx={{ display: 'flex' }}>
      <StyledAppBar position="fixed" open={drawerOpen}>
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            onClick={handleDrawerOpen}
            edge="start"
            sx={{ mr: 2, ...(drawerOpen && { display: 'none' }) }}
          >
            <MenuIcon />
          </IconButton>
          <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
            AI-Powered EDA Dashboard
          </Typography>
          {activeFileId && (
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <Typography variant="body2" sx={{ mr: 2 }}>
                Active File: {fileName}
              </Typography>
              <Button 
                variant="outlined" 
                color="inherit" 
                size="small" 
                startIcon={<DownloadIcon />}
                onClick={() => downloadFile(false)}
              >
                Original
              </Button>
              {cleanedFileId && (
                <Button 
                  variant="outlined" 
                  color="inherit" 
                  size="small" 
                  startIcon={<DownloadIcon />}
                  onClick={() => downloadFile(true)}
                  sx={{ ml: 1 }}
                >
                  Cleaned
                </Button>
              )}
            </Box>
          )}
        </Toolbar>
      </StyledAppBar>
      
      <Drawer
        sx={{
          width: drawerWidth,
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: drawerWidth,
            boxSizing: 'border-box',
          },
        }}
        variant="persistent"
        anchor="left"
        open={drawerOpen}
      >
        <DrawerHeader>
          <Typography variant="h6" sx={{ flexGrow: 1, ml: 2 }}>
            EDA Steps
          </Typography>
          <IconButton onClick={handleDrawerClose}>
            <ChevronLeftIcon />
          </IconButton>
        </DrawerHeader>
        <Divider />
        <List>
          {steps.map((text, index) => (
            <ListItem 
              button 
              key={text} 
              onClick={() => handleStepChange(index)}
              selected={activeStep === index}
              disabled={index > 0 && !activeFileId} // Disable steps if no file is uploaded
            >
              <ListItemIcon>
                {index === 0 && <UploadFileIcon />}
                {index === 1 && <TableChartIcon />}
                {index === 2 && <WarningIcon />}
                {index === 3 && <CleaningServicesIcon />}
                {index === 4 && <BarChartIcon />}
                {index === 5 && <WarningIcon />}
                {index === 6 && <BubbleChartIcon />}
              </ListItemIcon>
              <ListItemText primary={text} />
            </ListItem>
          ))}
        </List>
        <Divider />
        <Box sx={{ p: 2 }}>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Connection Status:
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <Box
              sx={{
                width: 10,
                height: 10,
                borderRadius: '50%',
                bgcolor: isConnected ? 'success.main' : 'error.main',
                mr: 1,
              }}
            />
            <Typography variant="body2">
              {isConnected ? 'Connected' : 'Disconnected'}
            </Typography>
          </Box>
        </Box>
      </Drawer>
      
      <Main open={drawerOpen}>
        <DrawerHeader />
        
        <Container maxWidth="lg">
          {error && (
            <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
              {error}
            </Alert>
          )}
          
          <Paper sx={{ p: 2, mb: 3 }}>
            <Stepper activeStep={activeStep} alternativeLabel>
              {steps.map((label, index) => (
                <Step key={label}>
                  <StepLabel>{label}</StepLabel>
                </Step>
              ))}
            </Stepper>
          </Paper>
          
          <Box sx={{ mb: 2 }}>
            {renderStepContent()}
          </Box>
          
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 2 }}>
            <Button
              variant="outlined"
              onClick={handleBack}
              disabled={activeStep === 0}
              startIcon={<ChevronLeftIcon />}
            >
              Back
            </Button>
            <Button
              variant="contained"
              onClick={handleNext}
              disabled={activeStep === steps.length - 1 || !activeFileId}
              endIcon={<ChevronRightIcon />}
            >
              Next
            </Button>
          </Box>
        </Container>
      </Main>
    </Box>
  );
};

export default Dashboard;