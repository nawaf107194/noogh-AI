import { useState, useEffect } from 'react';
import { StatusCard } from '../components/StatusCard';
import { MinisterCard } from '../components/MinisterCard';
import { useApi } from '../hooks/useApi';

interface SystemData {
  message?: string;
  version?: string;
  government?: { ministers: number; sectors: string[] };
}

export function Home() {
  const [currentTime, setCurrentTime] = useState(new Date());
  const { data, loading } = useApi<SystemData>('/api');

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const ministers = [
    { name: 'Knowledge Minister', role: 'Education & Research', status: 'active' as const, tasksCompleted: 142, efficiency: 94 },
    { name: 'Security Minister', role: 'System Security', status: 'busy' as const, tasksCompleted: 89, efficiency: 87 },
    { name: 'Finance Minister', role: 'Trading & Finance', status: 'active' as const, tasksCompleted: 156, efficiency: 91 },
    { name: 'Development Minister', role: 'Code & Features', status: 'idle' as const, tasksCompleted: 67, efficiency: 72 },
    { name: 'AI Core Minister', role: 'AI Operations', status: 'busy' as const, tasksCompleted: 203, efficiency: 96 },
    { name: 'Communication Minister', role: 'User Interaction', status: 'active' as const, tasksCompleted: 178, efficiency: 89 },
  ];

  return (
    <div className="p-8">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold gradient-text mb-2">Dashboard Overview</h1>
          <p className="text-gray-400">نظام نوغ الموحد للذكاء الاصطناعي</p>
        </div>
        <div className="text-right">
          <p className="text-gray-400 text-sm">System Time</p>
          <p className="text-white font-mono text-lg">{currentTime.toLocaleTimeString()}</p>
        </div>
      </div>

      <section className="mb-8">
        <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
          <span className="w-1 h-6 bg-gradient-to-b from-blue-500 to-purple-600 rounded-full"></span>
          System Overview
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <StatusCard
            title="API Status"
            value={loading ? '...' : data?.message ? 'Online' : 'Offline'}
            subtitle={data?.version || 'v1.0.0'}
            icon={<span className="text-white text-2xl">🚀</span>}
            gradient="from-green-500 to-emerald-600"
            pulse={true}
            trend="up"
            trendValue="99.9%"
          />
          <StatusCard
            title="Active Ministers"
            value={data?.government?.ministers || 14}
            subtitle="Government System"
            icon={<span className="text-white text-2xl">🏛️</span>}
            gradient="from-blue-500 to-cyan-600"
            trend="neutral"
            trendValue="14"
          />
          <StatusCard
            title="Total Requests"
            value="12.4K"
            subtitle="Last 24 hours"
            icon={<span className="text-white text-2xl">📊</span>}
            gradient="from-purple-500 to-pink-600"
            trend="up"
            trendValue="+23%"
          />
          <StatusCard
            title="System Health"
            value="98.5%"
            subtitle="All systems operational"
            icon={<span className="text-white text-2xl">💚</span>}
            gradient="from-orange-500 to-red-600"
            pulse={true}
            trend="up"
            trendValue="+2.1%"
          />
        </div>
      </section>

      <section className="mb-8">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <span className="w-1 h-6 bg-gradient-to-b from-blue-500 to-purple-600 rounded-full"></span>
            Government Ministers
          </h2>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {ministers.slice(0, 6).map((minister, index) => (
            <MinisterCard key={index} {...minister} />
          ))}
        </div>
      </section>

      <section>
        <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
          <span className="w-1 h-6 bg-gradient-to-b from-blue-500 to-purple-600 rounded-full"></span>
          Performance Metrics
        </h2>
        <div className="glass rounded-2xl p-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {['CPU', 'Memory', 'Disk'].map((metric, i) => (
              <div key={metric}>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-gray-400 text-sm">{metric} Usage</span>
                  <span className="text-white font-bold">{[45, 62, 38][i]}%</span>
                </div>
                <div className="w-full bg-slate-700 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full bg-gradient-to-r ${
                      ['from-blue-500 to-cyan-500', 'from-purple-500 to-pink-500', 'from-green-500 to-emerald-500'][i]
                    }`}
                    style={{ width: `${[45, 62, 38][i]}%` }}
                  ></div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}
