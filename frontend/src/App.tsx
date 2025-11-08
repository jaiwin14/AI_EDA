import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Toaster } from 'react-hot-toast';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import Upload from './pages/Upload';
import EDA from './pages/EDA';
import MLWorkflow from './pages/MLWorkflow';
// import Models from './pages/Models';
// import Predictions from './pages/Predictions';
import Preprocessing from './pages/Preprocessing';
import './App.css';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 2,
      staleTime: 5 * 60 * 1000, // 5 minutes
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <div className="App">
          <Navbar />
          <main className="main-content">
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/upload" element={<Upload />} />
              <Route path="/eda/:datasetId?" element={<EDA />} />
              {/* <Route path="/models/:datasetId?" element={<MLWorkflow />} />
              <Route path="/predictions/:modelId?" element={<MLWorkflow />} /> */}
              <Route path="/ml-workflow/:datasetId?" element={<MLWorkflow />} />
              <Route path="/preprocessing/:datasetId?" element={<Preprocessing />} />
              {/* <Route path="/models/:datasetId?" element={<Models />} />
              <Route path="/predictions/:modelId?" element={<Predictions />} /> */}
            </Routes>
          </main>
          <Toaster
            position="top-right"
            toastOptions={{
              duration: 4000,
              style: {
                background: '#363636',
                color: '#fff',
              },
            }}
          />
        </div>
      </Router>
    </QueryClientProvider>
  );
}

export default App;
