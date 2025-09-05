'use client';

import dynamic from 'next/dynamic';

// Dynamic imports to avoid SSR issues with WebSocket
const Dashboard = dynamic(() => import('@/components/Dashboard'), { ssr: false });
const Header = dynamic(() => import('@/components/Header'), { ssr: false });

const Home = () => {
  return (
    <div className="flex flex-col min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      <Header />
      <div className="pt-16">
        <main className="flex flex-1 overflow-hidden">
          <Dashboard />
        </main>
      </div>
    </div>
  );
};

export default Home;
