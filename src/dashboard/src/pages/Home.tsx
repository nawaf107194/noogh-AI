import { useState, useEffect } from 'react';
import { StatusCard } from '../components/StatusCard';
import { MinisterCard } from '../components/MinisterCard';
import { useApi } from '../hooks/useApi';

interface SystemOverview {
  success: boolean;
  data: {
    timestamp: string;
    overall_status: string;
    health_percent: number;
    active_components: number;
    total_components: number;
    mcp_server: {
      status: string;
      port: number;
      version: string;
      tools: number;
      resources: number;
    };
    brain_v4: {
      status: string;
      version: string;
      session_memories: number;
      capacity: number;
      usage_percent: number;
    };
    knowledge_index: {
      status: string;
      total_chunks: number;
      categories: string[];
      progress_percent: number;
    };
    daily_training: {
      status: string;
      latest_run: string | null;
      tasks_completed: number;
      success_rate: string;
    };
    ministers: {
      total: number;
      active: number;
      list: Array<{
        id: number;
        name: string;
        status: string;
        domain: string;
      }>;
    };
    automation_level: string;
  };
}

export function Home() {
  const [currentTime, setCurrentTime] = useState(new Date());
  const { data, loading } = useApi<SystemOverview>('/api/system/overview', 10000);

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const systemData = data?.data;
  const ministers = systemData?.ministers?.list || [];

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
            title="MCP Server"
            value={loading ? '...' : systemData?.mcp_server?.status === 'active' ? 'Online' : 'Offline'}
            subtitle={`v${systemData?.mcp_server?.version || '2.0'} • Port ${systemData?.mcp_server?.port || 8001}`}
            icon={<span className="text-white text-2xl">🔧</span>}
            gradient="from-green-500 to-emerald-600"
            pulse={systemData?.mcp_server?.status === 'active'}
            trend={systemData?.mcp_server?.status === 'active' ? 'up' : 'neutral'}
            trendValue={`${systemData?.mcp_server?.tools || 8} tools`}
          />
          <StatusCard
            title="Active Ministers"
            value={loading ? '...' : systemData?.ministers?.active || 0}
            subtitle={`${systemData?.ministers?.total || 14} total ministers`}
            icon={<span className="text-white text-2xl">🏛️</span>}
            gradient="from-blue-500 to-cyan-600"
            trend="neutral"
            trendValue={`${systemData?.ministers?.total || 14} ministers`}
          />
          <StatusCard
            title="Knowledge Chunks"
            value={loading ? '...' : systemData?.knowledge_index?.total_chunks || 0}
            subtitle={`${systemData?.knowledge_index?.categories?.length || 0} categories`}
            icon={<span className="text-white text-2xl">📚</span>}
            gradient="from-purple-500 to-pink-600"
            trend="up"
            trendValue={`${systemData?.knowledge_index?.progress_percent?.toFixed(0) || 0}%`}
          />
          <StatusCard
            title="System Health"
            value={loading ? '...' : `${systemData?.health_percent?.toFixed(1) || 0}%`}
            subtitle={systemData?.overall_status || 'Loading...'}
            icon={<span className="text-white text-2xl">💚</span>}
            gradient="from-orange-500 to-red-600"
            pulse={systemData?.health_percent === 100}
            trend={(systemData?.health_percent ?? 0) >= 80 ? 'up' : 'down'}
            trendValue={`${systemData?.active_components || 0}/${systemData?.total_components || 5} active`}
          />
        </div>
      </section>

      <section className="mb-8">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <span className="w-1 h-6 bg-gradient-to-b from-blue-500 to-purple-600 rounded-full"></span>
            Government Ministers ({systemData?.ministers?.active || 0}/{systemData?.ministers?.total || 14})
          </h2>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {ministers.slice(0, 6).map((minister) => (
            <MinisterCard
              key={minister.id}
              name={minister.name}
              role={minister.domain}
              status={minister.status as 'active' | 'busy' | 'idle'}
              tasksCompleted={Math.floor(Math.random() * 200) + 50}
              efficiency={Math.floor(Math.random() * 30) + 70}
            />
          ))}
        </div>
      </section>

      <section>
        <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
          <span className="w-1 h-6 bg-gradient-to-b from-blue-500 to-purple-600 rounded-full"></span>
          Core Components Status
        </h2>
        <div className="glass rounded-2xl p-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-gray-400 text-sm">Brain v4.0 Memory</span>
                <span className="text-white font-bold">{systemData?.brain_v4?.usage_percent?.toFixed(0) || 0}%</span>
              </div>
              <div className="w-full bg-slate-700 rounded-full h-2">
                <div
                  className="h-2 rounded-full bg-gradient-to-r from-blue-500 to-cyan-500"
                  style={{ width: `${systemData?.brain_v4?.usage_percent || 0}%` }}
                ></div>
              </div>
              <p className="text-gray-500 text-xs mt-1">
                {systemData?.brain_v4?.session_memories || 0} / {systemData?.brain_v4?.capacity || 100} memories
              </p>
            </div>
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-gray-400 text-sm">Knowledge Index</span>
                <span className="text-white font-bold">{systemData?.knowledge_index?.progress_percent?.toFixed(0) || 0}%</span>
              </div>
              <div className="w-full bg-slate-700 rounded-full h-2">
                <div
                  className="h-2 rounded-full bg-gradient-to-r from-purple-500 to-pink-500"
                  style={{ width: `${systemData?.knowledge_index?.progress_percent || 0}%` }}
                ></div>
              </div>
              <p className="text-gray-500 text-xs mt-1">
                {systemData?.knowledge_index?.total_chunks || 0} chunks indexed
              </p>
            </div>
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-gray-400 text-sm">Training Success</span>
                <span className="text-white font-bold">{systemData?.daily_training?.success_rate || '0%'}</span>
              </div>
              <div className="w-full bg-slate-700 rounded-full h-2">
                <div
                  className="h-2 rounded-full bg-gradient-to-r from-green-500 to-emerald-500"
                  style={{ width: systemData?.daily_training?.success_rate || '0%' }}
                ></div>
              </div>
              <p className="text-gray-500 text-xs mt-1">
                {systemData?.daily_training?.tasks_completed || 0} tasks completed
              </p>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
