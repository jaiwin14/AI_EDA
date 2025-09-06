'use client';

import React, { useState, useEffect } from 'react';
import {
  Box,
  Container,
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
  ListItemButton,
  Divider,
  Alert,
  Stepper,
  Step,
  StepLabel,
  IconButton,
  Chip,
  CircularProgress,
  Snackbar,
} from '@mui/material';
import { styled, Theme } from '@mui/material/styles';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import TableChartIcon from '@mui/icons-material/TableChart';
import WarningIcon from '@mui/icons-material/Warning';
import MenuIcon from '@mui/icons-material/Menu';
import ChevronLeftIcon from '@mui/icons-material/ChevronLeft';
import ChevronRightIcon from '@mui/icons-material/ChevronRight';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import FileUpload from './FileUpload';
import DataSummary from './DataSummary';
import MissingValues from './MissingValues';
import MissingValuesTreatment from './MissingValuesTreatment';
import DataDistribution from './DataDistribution';
import OutlierDetection from './OutlierDetection';
import Correlation from './Correlation';

const drawerWidth = 280;

const Main = styled('main', { shouldForwardProp: (prop) => prop !== 'open' })<{ open?: boolean }>(
  ({ theme, open }) => ({
    flexGrow: 1,
    padding: theme.spacing(3),
    paddingTop: `calc(${theme.spacing(3)} + 64px)`, // 64px is the header height
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
})<{ open?: boolean }>(({ theme, open }) => ({
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
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

export default function Dashboard() {
  const [activeStep, setActiveStep] = useState(0);
  const [drawerOpen, setDrawerOpen] = useState(true);
  const [activeFile, setActiveFile] = useState<File | null>(null);
  const [activeFileId, setActiveFileId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [snackbar, setSnackbar] = useState<{
    open: boolean;
    message: string;
    severity: 'success' | 'error' | 'info';
  }>({ open: false, message: '', severity: 'info' });

  // Test backend connection on component mount
  useEffect(() => {
    testBackendConnection();
  }, []);

  const testBackendConnection = async () => {
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/functions`);
      setIsConnected(response.ok);
    } catch (error) {
      setIsConnected(false);
      console.error('Backend connection failed:', error);
    }
  };

  const handleDrawerToggle = () => {
    setDrawerOpen(!drawerOpen);
  };

  const handleFileUploaded = (file: File, fileId: string) => {
    setActiveFile(file);
    setActiveFileId(fileId);
    setActiveStep(1);
    setError(null);
    showSnackbar('File uploaded successfully!', 'success');
  };

  const handleNext = () => {
    if (activeStep < steps.length - 1) {
      setActiveStep(activeStep + 1);
    }
  };

  const handleBack = () => {
    if (activeStep > 0) {
      setActiveStep(activeStep - 1);
    }
  };

  const handleStepChange = (step: number) => {
    if (step === 0 || (activeFileId && step <= activeStep)) {
      setActiveStep(step);
    }
  };

  const showSnackbar = (message: string, severity: 'success' | 'error' | 'info') => {
    setSnackbar({ open: true, message, severity });
  };

  const handleSnackbarClose = () => {
    setSnackbar({ ...snackbar, open: false });
  };

  const renderStepContent = () => {
    if (!isConnected) {
      return (
        <Box sx={{ textAlign: 'center', py: 8 }}>
          <ErrorIcon sx={{ fontSize: 64, color: 'error.main', mb: 2 }} />
          <Typography variant="h5" gutterBottom>
            Backend Connection Failed
          </Typography>
          <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
            Unable to connect to the backend server. Please make sure the backend is running.
          </Typography>
          <Button
            variant="contained"
            onClick={testBackendConnection}
            sx={{ mr: 2 }}
          >
            Retry Connection
          </Button>
          <Button
            variant="outlined"
            onClick={() => window.open('http://localhost:8000/docs', '_blank')}
          >
            Open API Docs
          </Button>
        </Box>
      );
    }

    switch (activeStep) {
      case 0:
        return (
          <Box sx={{ textAlign: 'center', py: 4 }}>
            <Typography variant="h4" gutterBottom sx={{ mb: 4 }}>
              Welcome to AI EDA Dashboard
            </Typography>
            <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
              Upload your data file to begin exploratory data analysis
            </Typography>
            <FileUpload onFileUploaded={handleFileUploaded} />
          </Box>
        );
      case 1:
        return activeFileId ? (
          <DataSummary activeFileId={activeFileId} summaryData={null} websocket={null} runAnalysis={() => {}} />
        ) : (
          <Box sx={{ textAlign: 'center', py: 4 }}>
            <Typography variant="h6" color="text.secondary">
              Please upload a file first
            </Typography>
          </Box>
        );
      case 2:
        return activeFileId ? (
          <MissingValues activeFileId={activeFileId} missingData={null} websocket={null} runAnalysis={() => {}} />
        ) : (
          <Box sx={{ textAlign: 'center', py: 4 }}>
            <Typography variant="h6" color="text.secondary">
              Please upload a file first
            </Typography>
          </Box>
        );
      case 3:
        return activeFileId ? (
          <MissingValuesTreatment activeFileId={activeFileId} treatmentData={null} websocket={null} onTreatmentApplied={() => {}} />
        ) : (
          <Box sx={{ textAlign: 'center', py: 4 }}>
            <Typography variant="h6" color="text.secondary">
              Please upload a file first
            </Typography>
          </Box>
        );
      case 4:
        return activeFileId ? (
          <DataDistribution activeFileId={activeFileId} distributionData={null} websocket={null} runAnalysis={() => {}} />
        ) : (
          <Box sx={{ textAlign: 'center', py: 4 }}>
            <Typography variant="h6" color="text.secondary">
              Please upload a file first
            </Typography>
          </Box>
        );
      case 5:
        return activeFileId ? (
          <OutlierDetection activeFileId={activeFileId} outlierData={null} websocket={null} runAnalysis={() => {}} />
        ) : (
          <Box sx={{ textAlign: 'center', py: 4 }}>
            <Typography variant="h6" color="text.secondary">
              Please upload a file first
            </Typography>
          </Box>
        );
      case 6:
        return activeFileId ? (
          <Correlation activeFileId={activeFileId} correlationData={null} websocket={null} runAnalysis={() => {}} />
        ) : (
          <Box sx={{ textAlign: 'center', py: 4 }}>
            <Typography variant="h6" color="text.secondary">
              Please upload a file first
            </Typography>
          </Box>
        );
      default:
        return null;
    }
  };

  return (
    <Box sx={{ display: 'flex', height: '100vh' }}>
      <StyledAppBar position="fixed" open={drawerOpen}>
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            onClick={handleDrawerToggle}
            edge="start"
            sx={{ mr: 2 }}
          >
            <MenuIcon />
          </IconButton>
          <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
            AI EDA Dashboard
          </Typography>
          <Chip
            icon={isConnected ? <CheckCircleIcon /> : <ErrorIcon />}
            label={isConnected ? 'Connected' : 'Disconnected'}
            color={isConnected ? 'success' : 'error'}
            variant="outlined"
            sx={{ color: 'white', borderColor: 'white' }}
          />
        </Toolbar>
      </StyledAppBar>

      <Drawer
        sx={{
          width: drawerWidth,
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: drawerWidth,
            boxSizing: 'border-box',
            background: 'linear-gradient(180deg, #1e293b 0%, #334155 100%)',
            color: 'white',
            borderRight: '1px solid rgba(255, 255, 255, 0.1)',
          },
        }}
        variant="persistent"
        anchor="left"
        open={drawerOpen}
      >
        <DrawerHeader>
          <Typography variant="h6" sx={{ flexGrow: 1, ml: 2, color: 'white', fontWeight: 600 }}>
            EDA Steps
          </Typography>
          <IconButton 
            onClick={handleDrawerToggle}
            sx={{ 
              color: 'white',
              '&:hover': {
                backgroundColor: 'rgba(255, 255, 255, 0.1)',
              }
            }}
          >
            <ChevronLeftIcon />
          </IconButton>
        </DrawerHeader>
        <Divider sx={{ borderColor: 'rgba(255, 255, 255, 0.1)' }} />
        <List>
          {steps.map((text, index) => (
            <ListItemButton
              key={text} 
              onClick={() => handleStepChange(index)}
              divider
              disabled={index > 0 && !activeFileId}
              selected={activeStep === index}
              sx={{
                color: 'white',
                '&:hover': {
                  backgroundColor: 'rgba(255, 255, 255, 0.1)',
                },
                '&.Mui-selected': {
                  backgroundColor: 'rgba(59, 130, 246, 0.2)',
                  borderRight: '3px solid #3B82F6',
                  '&:hover': {
                    backgroundColor: 'rgba(59, 130, 246, 0.3)',
                  }
                },
                '&.Mui-disabled': {
                  color: 'rgba(255, 255, 255, 0.3)',
                }
              }}
            >
              <ListItemIcon sx={{ color: 'white' }}>
                {index === 0 && <UploadFileIcon />}
                {index === 1 && <TableChartIcon />}
                {index === 2 && <WarningIcon />}
                {index === 3 && <WarningIcon />}
                {index === 4 && <TableChartIcon />}
                {index === 5 && <WarningIcon />}
                {index === 6 && <TableChartIcon />}
              </ListItemIcon>
              <ListItemText primary={text} />
            </ListItemButton>
          ))}
        </List>
        <Divider sx={{ borderColor: 'rgba(255, 255, 255, 0.1)' }} />
        <Box sx={{ p: 2, borderTop: '1px solid rgba(255, 255, 255, 0.1)' }}>
          <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.7)', mb: 1 }}>
            Connection Status:
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <Box
              sx={{
                width: 8,
                height: 8,
                borderRadius: '50%',
                bgcolor: isConnected ? '#10B981' : '#EF4444',
                mr: 1,
                boxShadow: isConnected ? '0 0 8px rgba(16, 185, 129, 0.5)' : '0 0 8px rgba(239, 68, 68, 0.5)',
              }}
            />
            <Typography variant="body2" sx={{ color: 'white', fontSize: '0.875rem' }}>
              {isConnected ? 'Connected' : 'Disconnected'}
            </Typography>
          </Box>
        </Box>
      </Drawer>

      <Main open={drawerOpen}>
        <DrawerHeader />
        
        <Container maxWidth="lg" sx={{ py: 3 }}>
          {error && (
            <Alert 
              severity="error" 
              sx={{ 
                mb: 3,
                borderRadius: 2,
                border: '1px solid rgba(239, 68, 68, 0.2)',
                '& .MuiAlert-icon': {
                  color: '#EF4444',
                }
              }} 
              onClose={() => setError(null)}
            >
              {error}
            </Alert>
          )}
          
          <Paper sx={{ 
            p: 3, 
            mb: 3, 
            background: 'linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%)',
            borderRadius: 3,
            boxShadow: '0 10px 25px rgba(0, 0, 0, 0.1)',
            border: '1px solid rgba(59, 130, 246, 0.1)'
          }}>
            <Stepper activeStep={activeStep} alternativeLabel>
              {steps.map((label, index) => (
                <Step key={label}>
                  <StepLabel 
                    sx={{
                      '& .MuiStepLabel-label': {
                        color: index <= activeStep ? '#3B82F6' : '#64748B',
                        fontWeight: index <= activeStep ? 600 : 400,
                      },
                      '& .MuiStepIcon-root': {
                        color: index <= activeStep ? '#3B82F6' : '#CBD5E1',
                      }
                    }}
                  >
                    {label}
                  </StepLabel>
                </Step>
              ))}
            </Stepper>
          </Paper>
          
          <Box sx={{ mb: 2 }}>
            {renderStepContent()}
          </Box>
          
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 3 }}>
            <Button
              variant="outlined"
              onClick={handleBack}
              disabled={activeStep === 0}
              startIcon={<ChevronLeftIcon />}
              sx={{
                borderRadius: 2,
                px: 3,
                py: 1.5,
                borderColor: '#3B82F6',
                color: '#3B82F6',
                '&:hover': {
                  borderColor: '#2563EB',
                  backgroundColor: 'rgba(59, 130, 246, 0.04)',
                }
              }}
            >
              Back
            </Button>
            <Button
              variant="contained"
              onClick={handleNext}
              disabled={activeStep === steps.length - 1 || !activeFileId}
              endIcon={<ChevronRightIcon />}
              sx={{
                borderRadius: 2,
                px: 3,
                py: 1.5,
                background: 'linear-gradient(135deg, #3B82F6 0%, #2563EB 100%)',
                '&:hover': {
                  background: 'linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%)',
                },
                '&:disabled': {
                  background: '#CBD5E1',
                }
              }}
            >
              Next
            </Button>
          </Box>
        </Container>
      </Main>

      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={handleSnackbarClose}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert 
          onClose={handleSnackbarClose} 
          severity={snackbar.severity}
          sx={{ width: '100%' }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
}