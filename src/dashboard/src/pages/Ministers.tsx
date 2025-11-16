import { MinisterCard } from '../components/MinisterCard';
import { Users, TrendingUp, Award } from 'lucide-react';

export function Ministers() {
  const allMinisters = [
    { name: 'Knowledge Minister', role: 'Education & Research', status: 'active' as const, tasksCompleted: 142, efficiency: 94 },
    { name: 'Security Minister', role: 'System Security', status: 'busy' as const, tasksCompleted: 89, efficiency: 87 },
    { name: 'Finance Minister', role: 'Trading & Finance', status: 'active' as const, tasksCompleted: 156, efficiency: 91 },
    { name: 'Development Minister', role: 'Code & Features', status: 'idle' as const, tasksCompleted: 67, efficiency: 72 },
    { name: 'AI Core Minister', role: 'AI Operations', status: 'busy' as const, tasksCompleted: 203, efficiency: 96 },
    { name: 'Communication Minister', role: 'User Interaction', status: 'active' as const, tasksCompleted: 178, efficiency: 89 },
    { name: 'Analysis Minister', role: 'Data Analytics', status: 'active' as const, tasksCompleted: 134, efficiency: 88 },
    { name: 'Strategy Minister', role: 'Planning & Strategy', status: 'busy' as const, tasksCompleted: 98, efficiency: 85 },
    { name: 'Training Minister', role: 'Model Training', status: 'active' as const, tasksCompleted: 187, efficiency: 92 },
    { name: 'Reasoning Minister', role: 'Logic & Reasoning', status: 'busy' as const, tasksCompleted: 145, efficiency: 90 },
    { name: 'Privacy Minister', role: 'Data Privacy', status: 'active' as const, tasksCompleted: 112, efficiency: 86 },
    { name: 'Creativity Minister', role: 'Creative Solutions', status: 'idle' as const, tasksCompleted: 76, efficiency: 78 },
    { name: 'Resources Minister', role: 'Resource Management', status: 'active' as const, tasksCompleted: 121, efficiency: 84 },
    { name: 'Integration Minister', role: 'System Integration', status: 'busy' as const, tasksCompleted: 165, efficiency: 93 },
  ];

  const avgEfficiency = Math.round(allMinisters.reduce((acc, m) => acc + m.efficiency, 0) / allMinisters.length);
  const totalTasks = allMinisters.reduce((acc, m) => acc + m.tasksCompleted, 0);
  const activeCount = allMinisters.filter(m => m.status === 'active').length;

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold gradient-text mb-2">Government Ministers</h1>
        <p className="text-gray-400">Full overview of all 14 ministers and their performance</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <div className="glass rounded-xl p-6">
          <div className="flex items-center gap-4">
            <div className="p-3 rounded-lg bg-gradient-to-br from-blue-500 to-cyan-600">
              <Users className="text-white" size={24} />
            </div>
            <div>
              <p className="text-gray-400 text-sm">Active Ministers</p>
              <p className="text-2xl font-bold text-white">{activeCount}/14</p>
            </div>
          </div>
        </div>
        <div className="glass rounded-xl p-6">
          <div className="flex items-center gap-4">
            <div className="p-3 rounded-lg bg-gradient-to-br from-purple-500 to-pink-600">
              <TrendingUp className="text-white" size={24} />
            </div>
            <div>
              <p className="text-gray-400 text-sm">Total Tasks</p>
              <p className="text-2xl font-bold text-white">{totalTasks}</p>
            </div>
          </div>
        </div>
        <div className="glass rounded-xl p-6">
          <div className="flex items-center gap-4">
            <div className="p-3 rounded-lg bg-gradient-to-br from-green-500 to-emerald-600">
              <Award className="text-white" size={24} />
            </div>
            <div>
              <p className="text-gray-400 text-sm">Avg Efficiency</p>
              <p className="text-2xl font-bold text-white">{avgEfficiency}%</p>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
        {allMinisters.map((minister, index) => (
          <MinisterCard key={index} {...minister} />
        ))}
      </div>
    </div>
  );
}
