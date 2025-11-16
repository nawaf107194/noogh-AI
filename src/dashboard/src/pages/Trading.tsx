import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { TrendingUp, TrendingDown, DollarSign, Activity } from 'lucide-react';

const priceData = [
  { time: '00:00', BTC: 42000, ETH: 2200, profit: 1200 },
  { time: '04:00', BTC: 43500, ETH: 2350, profit: 2400 },
  { time: '08:00', BTC: 42800, ETH: 2180, profit: 1800 },
  { time: '12:00', BTC: 44200, ETH: 2420, profit: 3100 },
  { time: '16:00', BTC: 45100, ETH: 2550, profit: 4200 },
  { time: '20:00', BTC: 44800, ETH: 2480, profit: 3800 },
  { time: '24:00', BTC: 46200, ETH: 2650, profit: 5100 },
];

export function Trading() {
  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold gradient-text mb-2">Trading Dashboard</h1>
        <p className="text-gray-400">Real-time cryptocurrency trading analytics</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        {[
          { icon: DollarSign, label: 'Portfolio Value', value: '$125,480', change: '+12.5%', trend: 'up', color: 'from-green-500 to-emerald-600' },
          { icon: TrendingUp, label: 'Total Profit', value: '$15,280', change: '+8.3%', trend: 'up', color: 'from-blue-500 to-cyan-600' },
          { icon: Activity, label: 'Active Trades', value: '24', change: '+4', trend: 'up', color: 'from-purple-500 to-pink-600' },
          { icon: TrendingDown, label: 'Win Rate', value: '68%', change: '+2.1%', trend: 'up', color: 'from-orange-500 to-red-600' },
        ].map((stat, i) => (
          <div key={i} className="glass rounded-xl p-6">
            <div className="flex items-start justify-between mb-4">
              <div className={`p-3 rounded-lg bg-gradient-to-br ${stat.color}`}>
                <stat.icon className="text-white" size={24} />
              </div>
              <span className={`text-sm font-semibold ${stat.trend === 'up' ? 'text-green-400' : 'text-red-400'}`}>
                {stat.change}
              </span>
            </div>
            <p className="text-gray-400 text-sm mb-1">{stat.label}</p>
            <p className="text-2xl font-bold text-white">{stat.value}</p>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <div className="glass rounded-2xl p-6">
          <h3 className="text-lg font-bold text-white mb-4">Bitcoin Price (24h)</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={priceData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="time" stroke="#9CA3AF" />
              <YAxis stroke="#9CA3AF" />
              <Tooltip contentStyle={{ backgroundColor: '#1F2937', border: 'none', borderRadius: '8px' }} />
              <Line type="monotone" dataKey="BTC" stroke="#3B82F6" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="glass rounded-2xl p-6">
          <h3 className="text-lg font-bold text-white mb-4">Ethereum Price (24h)</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={priceData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="time" stroke="#9CA3AF" />
              <YAxis stroke="#9CA3AF" />
              <Tooltip contentStyle={{ backgroundColor: '#1F2937', border: 'none', borderRadius: '8px' }} />
              <Line type="monotone" dataKey="ETH" stroke="#8B5CF6" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="glass rounded-2xl p-6">
        <h3 className="text-lg font-bold text-white mb-4">Cumulative Profit</h3>
        <ResponsiveContainer width="100%" height={300}>
          <AreaChart data={priceData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="time" stroke="#9CA3AF" />
            <YAxis stroke="#9CA3AF" />
            <Tooltip contentStyle={{ backgroundColor: '#1F2937', border: 'none', borderRadius: '8px' }} />
            <Area type="monotone" dataKey="profit" stroke="#10B981" fill="url(#profitGradient)" />
            <defs>
              <linearGradient id="profitGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#10B981" stopOpacity={0.8} />
                <stop offset="95%" stopColor="#10B981" stopOpacity={0} />
              </linearGradient>
            </defs>
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
