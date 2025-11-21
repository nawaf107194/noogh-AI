import { BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { FileText, Download } from 'lucide-react';

const ministerPerformance = [
  { name: 'Knowledge', tasks: 142, efficiency: 94 },
  { name: 'Security', tasks: 89, efficiency: 87 },
  { name: 'Finance', tasks: 156, efficiency: 91 },
  { name: 'Development', tasks: 67, efficiency: 72 },
  { name: 'AI Core', tasks: 203, efficiency: 96 },
  { name: 'Communication', tasks: 178, efficiency: 89 },
];

const sectorDistribution = [
  { name: 'Knowledge', value: 21 },
  { name: 'Security', value: 14 },
  { name: 'Development', value: 14 },
  { name: 'Analysis', value: 14 },
  { name: 'AI Core', value: 14 },
  { name: 'Communication', value: 14 },
  { name: 'Finance', value: 7 },
];

const COLORS = ['#3B82F6', '#8B5CF6', '#10B981', '#F59E0B', '#EF4444', '#06B6D4', '#EC4899'];

export function Reports() {
  return (
    <div className="p-8">
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold gradient-text mb-2">Reports & Analytics</h1>
          <p className="text-gray-400">Comprehensive system performance reports</p>
        </div>
        <button className="flex items-center gap-2 px-4 py-2 rounded-lg bg-gradient-to-r from-blue-500 to-purple-600 text-white hover:shadow-lg transition-all">
          <Download size={20} />
          Export Report
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <div className="glass rounded-2xl p-6">
          <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
            <FileText size={20} className="text-blue-500" />
            Minister Performance
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={ministerPerformance}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="name" stroke="#9CA3AF" />
              <YAxis stroke="#9CA3AF" />
              <Tooltip contentStyle={{ backgroundColor: '#1F2937', border: 'none', borderRadius: '8px' }} />
              <Legend />
              <Bar dataKey="tasks" fill="#3B82F6" radius={[8, 8, 0, 0]} />
              <Bar dataKey="efficiency" fill="#8B5CF6" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="glass rounded-2xl p-6">
          <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
            <FileText size={20} className="text-purple-500" />
            Sector Distribution
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={sectorDistribution}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={(props: any) => `${props.name} ${(props.percent * 100).toFixed(0)}%`}
                outerRadius={100}
                fill="#8884d8"
                dataKey="value"
              >
                {sectorDistribution.map((_entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip contentStyle={{ backgroundColor: '#1F2937', border: 'none', borderRadius: '8px' }} />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="glass rounded-2xl p-6">
        <h3 className="text-lg font-bold text-white mb-4">Recent Activity</h3>
        <div className="space-y-3">
          {[
            { minister: 'AI Core Minister', action: 'Completed model training', time: '2 minutes ago', status: 'success' },
            { minister: 'Finance Minister', action: 'Executed 5 trades', time: '15 minutes ago', status: 'success' },
            { minister: 'Security Minister', action: 'Security scan completed', time: '1 hour ago', status: 'warning' },
            { minister: 'Development Minister', action: 'Code review pending', time: '2 hours ago', status: 'pending' },
            { minister: 'Knowledge Minister', action: 'Database updated', time: '3 hours ago', status: 'success' },
          ].map((activity, i) => (
            <div key={i} className="flex items-center justify-between p-4 bg-slate-800 bg-opacity-50 rounded-lg">
              <div className="flex items-center gap-4">
                <div className={`w-2 h-2 rounded-full ${
                  activity.status === 'success' ? 'bg-green-500' :
                  activity.status === 'warning' ? 'bg-yellow-500' : 'bg-blue-500'
                } pulse-dot`}></div>
                <div>
                  <p className="text-white font-medium">{activity.minister}</p>
                  <p className="text-gray-400 text-sm">{activity.action}</p>
                </div>
              </div>
              <span className="text-gray-500 text-sm">{activity.time}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
