interface MinisterCardProps {
  name: string;
  role: string;
  status: 'active' | 'idle' | 'busy';
  tasksCompleted: number;
  efficiency: number;
}

export function MinisterCard({ name, role, status, tasksCompleted, efficiency }: MinisterCardProps) {
  const statusConfig = {
    active: {
      color: 'bg-green-500',
      text: 'Active',
      pulse: true
    },
    idle: {
      color: 'bg-yellow-500',
      text: 'Idle',
      pulse: false
    },
    busy: {
      color: 'bg-blue-500',
      text: 'Busy',
      pulse: true
    }
  };

  const config = statusConfig[status];
  const efficiencyColor = efficiency >= 80 ? 'text-green-400' : efficiency >= 60 ? 'text-yellow-400' : 'text-red-400';

  return (
    <div className="glass rounded-xl p-4 hover:scale-105 transition-all duration-300 hover:shadow-lg group cursor-pointer">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          <div className="relative">
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-white font-bold">
              {name.charAt(0)}
            </div>
            <span className={`absolute bottom-0 right-0 w-3 h-3 ${config.color} rounded-full border-2 border-slate-800 ${config.pulse ? 'pulse-dot' : ''}`}></span>
          </div>
          <div>
            <h4 className="text-white font-semibold text-sm">{name}</h4>
            <p className="text-gray-400 text-xs">{role}</p>
          </div>
        </div>
        <span className={`text-xs px-2 py-1 rounded-full ${config.color} bg-opacity-20 text-white`}>
          {config.text}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-3 mt-3">
        <div className="bg-slate-800 bg-opacity-50 rounded-lg p-2">
          <p className="text-gray-400 text-xs mb-1">Tasks</p>
          <p className="text-white font-bold text-sm">{tasksCompleted}</p>
        </div>
        <div className="bg-slate-800 bg-opacity-50 rounded-lg p-2">
          <p className="text-gray-400 text-xs mb-1">Efficiency</p>
          <p className={`font-bold text-sm ${efficiencyColor}`}>{efficiency}%</p>
        </div>
      </div>
    </div>
  );
}
