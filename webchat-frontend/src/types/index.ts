import { ReactNode } from 'react';

export interface FileUploadProps {
  onFileUploaded: (file: File) => void;
  children?: ReactNode;
}

export interface ChatInterfaceProps {
  file: File | null;
  onAnalysisRequest: (request: string) => void;
  analysisResults: AnalysisResult[];
  children?: ReactNode;
}

export interface SidebarProps {
  activeFileId: string | null;
  children?: ReactNode;
}

export interface AnalysisResult {
  type: string;
  data: any;
  timestamp: string;
}

export interface LayoutProps {
  children: ReactNode;
}
