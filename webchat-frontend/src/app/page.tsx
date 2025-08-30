'use client';

import dynamic from 'next/dynamic';

// Dynamic imports to avoid SSR issues with WebSocket
const Dashboard = dynamic(() => import('@/components/Dashboard'), { ssr: false });
const Header = dynamic(() => import('@/components/Header'), { ssr: false });

const Home = () => {
  return (
    <div className="flex flex-col h-screen bg-[#FFEAD8]">
      <Header />
      <main className="flex flex-1 overflow-hidden">
        <Dashboard />
        </main>
      </div>
    );
  };
  
  export default Home;
