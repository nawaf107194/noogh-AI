import { useState } from 'react';
import { Play, RefreshCw, Database, Brain, Server, Clock, HardDrive, CheckCircle, AlertCircle } from 'lucide-react';
import { useApi } from '../hooks/useApi';

interface AutomationStatus {
  success: boolean;
  timestamp: string;
  automation: {
    mcp_server: {
      status: string;
      port: number;
      version: string;
      tools: number;
      resources: number;
      features: string[];
    };
    brain_v4: {
      status: string;
      version: string;
      session_memories: number;
      capacity: number;
      features: string[];
    };
    knowledge_index: {
      status: string;
      version: string;
      total_chunks: number;
      categories: string[];
      target_achieved: string;
      progress: string;
    };
    daily_training: {
      status: string;
      cron_active: boolean;
      schedule: string;
      tasks_completed: number;
      latest_run: string;
      success_rate: string;
    };
  };
  overall_status: string;
  summary: {
    total_features: number;
    features_active: number;
    automation_level: string;
    manual_intervention_required: string;
  };
}

export function Automation() {
  const { data, loading, error } = useApi<AutomationStatus>('/api/automation/status', 10000);
  const [isTraining, setIsTraining] = useState(false);

  const handleManualTraining = async () => {
    setIsTraining(true);
    try {
      const response = await fetch('http://localhost:8000/api/automation/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      const result = await response.json();
      alert(result.success ? 'تم بدء التدريب بنجاح!' : 'فشل التدريب: ' + result.error);
    } catch (err) {
      alert('خطأ في الاتصال بالخادم');
    } finally {
      setIsTraining(false);
    }
  };

  if (loading && !data) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-8">
        <div className="glass rounded-2xl p-6 border border-red-500/20">
          <div className="flex items-center gap-3 text-red-400">
            <AlertCircle size={24} />
            <div>
              <h3 className="font-bold">خطأ في تحميل البيانات</h3>
              <p className="text-sm text-gray-400">{error}</p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  const automation = data?.automation;
  const summary = data?.summary;

  return (
    <div className="p-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold gradient-text mb-2">Automation Dashboard</h1>
          <p className="text-gray-400">لوحة تحكم النظام الآلي - {data?.overall_status}</p>
        </div>
        <button
          onClick={handleManualTraining}
          disabled={isTraining}
          className="flex items-center gap-2 px-6 py-3 rounded-xl bg-gradient-to-r from-blue-500 to-purple-600 text-white font-bold hover:shadow-lg hover:scale-105 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isTraining ? (
            <>
              <RefreshCw size={20} className="animate-spin" />
              جاري التدريب...
            </>
          ) : (
            <>
              <Play size={20} />
              تشغيل تدريب يدوي
            </>
          )}
        </button>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        <div className="glass rounded-2xl p-6">
          <div className="flex items-center justify-between mb-3">
            <CheckCircle className="text-green-500" size={32} />
            <span className="text-3xl font-bold gradient-text">{summary?.features_active}/{summary?.total_features}</span>
          </div>
          <h3 className="text-gray-400 text-sm mb-1">المكونات النشطة</h3>
          <p className="text-white font-medium">جميع الأنظمة تعمل</p>
        </div>

        <div className="glass rounded-2xl p-6">
          <div className="flex items-center justify-between mb-3">
            <Server className="text-blue-500" size={32} />
            <span className="text-3xl font-bold gradient-text">{summary?.automation_level}</span>
          </div>
          <h3 className="text-gray-400 text-sm mb-1">مستوى الأتمتة</h3>
          <p className="text-white font-medium">تشغيل كامل</p>
        </div>

        <div className="glass rounded-2xl p-6">
          <div className="flex items-center justify-between mb-3">
            <Database className="text-purple-500" size={32} />
            <span className="text-3xl font-bold gradient-text">{automation?.knowledge_index?.total_chunks}</span>
          </div>
          <h3 className="text-gray-400 text-sm mb-1">قطع المعرفة</h3>
          <p className="text-white font-medium">{automation?.knowledge_index?.progress}</p>
        </div>

        <div className="glass rounded-2xl p-6">
          <div className="flex items-center justify-between mb-3">
            <Brain className="text-pink-500" size={32} />
            <span className="text-3xl font-bold gradient-text">{automation?.brain_v4?.session_memories}/{automation?.brain_v4?.capacity}</span>
          </div>
          <h3 className="text-gray-400 text-sm mb-1">ذاكرة الدماغ</h3>
          <p className="text-white font-medium">جلسات نشطة</p>
        </div>
      </div>

      {/* Main Components */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        {/* MCP Server */}
        <div className="glass rounded-2xl p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <Server className="text-blue-500" size={24} />
              <h3 className="text-xl font-bold text-white">MCP Server</h3>
            </div>
            <span className="px-3 py-1 rounded-full bg-green-500/20 text-green-400 text-sm font-medium">
              {automation?.mcp_server?.status}
            </span>
          </div>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-400">Version</span>
              <span className="text-white font-medium">v{automation?.mcp_server?.version}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Port</span>
              <span className="text-white font-medium">{automation?.mcp_server?.port}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Tools</span>
              <span className="text-white font-medium">{automation?.mcp_server?.tools}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Resources</span>
              <span className="text-white font-medium">{automation?.mcp_server?.resources}</span>
            </div>
            <div className="mt-4 pt-4 border-t border-gray-700">
              <h4 className="text-gray-400 text-sm mb-2">Features:</h4>
              <div className="flex flex-wrap gap-2">
                {automation?.mcp_server?.features?.map((feature, i) => (
                  <span key={i} className="px-2 py-1 rounded-lg bg-blue-500/10 text-blue-400 text-xs">
                    {feature}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Brain v4.0 */}
        <div className="glass rounded-2xl p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <Brain className="text-pink-500" size={24} />
              <h3 className="text-xl font-bold text-white">Brain v4.0</h3>
            </div>
            <span className="px-3 py-1 rounded-full bg-green-500/20 text-green-400 text-sm font-medium">
              {automation?.brain_v4?.status}
            </span>
          </div>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-400">Version</span>
              <span className="text-white font-medium">v{automation?.brain_v4?.version}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Session Memories</span>
              <span className="text-white font-medium">{automation?.brain_v4?.session_memories}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Capacity</span>
              <span className="text-white font-medium">{automation?.brain_v4?.capacity}</span>
            </div>
            <div>
              <div className="flex justify-between mb-2">
                <span className="text-gray-400">Usage</span>
                <span className="text-white font-medium">
                  {automation?.brain_v4?.session_memories && automation?.brain_v4?.capacity
                    ? Math.round((automation.brain_v4.session_memories / automation.brain_v4.capacity) * 100)
                    : 0}%
                </span>
              </div>
              <div className="w-full bg-slate-700 rounded-full h-2">
                <div
                  className="h-2 rounded-full bg-gradient-to-r from-pink-500 to-purple-500"
                  style={{
                    width: `${automation?.brain_v4?.session_memories && automation?.brain_v4?.capacity
                      ? (automation.brain_v4.session_memories / automation.brain_v4.capacity) * 100
                      : 0}%`
                  }}
                ></div>
              </div>
            </div>
            <div className="mt-4 pt-4 border-t border-gray-700">
              <h4 className="text-gray-400 text-sm mb-2">Features:</h4>
              <div className="flex flex-wrap gap-2">
                {automation?.brain_v4?.features?.map((feature, i) => (
                  <span key={i} className="px-2 py-1 rounded-lg bg-pink-500/10 text-pink-400 text-xs">
                    {feature}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Knowledge Index & Training */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Knowledge Index */}
        <div className="glass rounded-2xl p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <Database className="text-purple-500" size={24} />
              <h3 className="text-xl font-bold text-white">Knowledge Index</h3>
            </div>
            <span className="px-3 py-1 rounded-full bg-green-500/20 text-green-400 text-sm font-medium">
              {automation?.knowledge_index?.status}
            </span>
          </div>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-400">Version</span>
              <span className="text-white font-medium">{automation?.knowledge_index?.version}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Total Chunks</span>
              <span className="text-white font-medium">{automation?.knowledge_index?.total_chunks}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Target</span>
              <span className="text-white font-medium">{automation?.knowledge_index?.target_achieved}</span>
            </div>
            <div>
              <div className="flex justify-between mb-2">
                <span className="text-gray-400">Progress</span>
                <span className="text-white font-medium">{automation?.knowledge_index?.progress}</span>
              </div>
              <div className="w-full bg-slate-700 rounded-full h-2">
                <div
                  className="h-2 rounded-full bg-gradient-to-r from-purple-500 to-blue-500"
                  style={{ width: automation?.knowledge_index?.progress || '0%' }}
                ></div>
              </div>
            </div>
            <div className="mt-4 pt-4 border-t border-gray-700">
              <h4 className="text-gray-400 text-sm mb-2">Categories ({automation?.knowledge_index?.categories?.length || 0}):</h4>
              <div className="flex flex-wrap gap-2">
                {automation?.knowledge_index?.categories?.map((category, i) => (
                  <span key={i} className="px-2 py-1 rounded-lg bg-purple-500/10 text-purple-400 text-xs">
                    {category}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Daily Training */}
        <div className="glass rounded-2xl p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <Clock className="text-green-500" size={24} />
              <h3 className="text-xl font-bold text-white">Daily Training</h3>
            </div>
            <span className="px-3 py-1 rounded-full bg-green-500/20 text-green-400 text-sm font-medium">
              {automation?.daily_training?.status}
            </span>
          </div>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-400">Schedule</span>
              <span className="text-white font-medium">{automation?.daily_training?.schedule}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Cron Job</span>
              <span className={`font-medium ${automation?.daily_training?.cron_active ? 'text-green-400' : 'text-red-400'}`}>
                {automation?.daily_training?.cron_active ? '✅ Active' : '❌ Inactive'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Success Rate</span>
              <span className="text-white font-medium">{automation?.daily_training?.success_rate}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Tasks Completed</span>
              <span className="text-white font-medium">{automation?.daily_training?.tasks_completed}</span>
            </div>
            <div className="mt-4 pt-4 border-t border-gray-700">
              <div className="flex justify-between text-sm">
                <span className="text-gray-400">Latest Run:</span>
                <span className="text-gray-300">
                  {automation?.daily_training?.latest_run
                    ? new Date(automation.daily_training.latest_run).toLocaleString('ar-SA')
                    : 'N/A'}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* System Info */}
      <div className="mt-6 glass rounded-2xl p-6">
        <div className="flex items-center gap-3 mb-4">
          <HardDrive className="text-cyan-500" size={24} />
          <h3 className="text-xl font-bold text-white">System Information</h3>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="space-y-2">
            <p className="text-gray-400 text-sm">Overall Status</p>
            <p className="text-2xl font-bold gradient-text">{data?.overall_status}</p>
          </div>
          <div className="space-y-2">
            <p className="text-gray-400 text-sm">Manual Intervention</p>
            <p className="text-2xl font-bold text-green-400">{summary?.manual_intervention_required}</p>
          </div>
          <div className="space-y-2">
            <p className="text-gray-400 text-sm">Last Update</p>
            <p className="text-lg font-medium text-gray-300">
              {data?.timestamp ? new Date(data.timestamp).toLocaleString('ar-SA') : 'N/A'}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
