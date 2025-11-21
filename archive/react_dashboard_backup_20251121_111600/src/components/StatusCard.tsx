import type { ReactNode } from 'react';

interface StatusCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon?: ReactNode;
  trend?: 'up' | 'down' | 'neutral';
  trendValue?: string;
  gradient?: string;
  pulse?: boolean;
}

export function StatusCard({
  title,
  value,
  subtitle,
  icon,
  trend,
  trendValue,
  gradient = 'from-blue-500 to-purple-600',
  pulse = false
}: StatusCardProps) {
  const trendColors = {
    up: 'text-green-400',
    down: 'text-red-400',
    neutral: 'text-gray-400',
  };

  const trendIcons = {
    up: '↑',
    down: '↓',
    neutral: '→',
  };

  return (
    <div className="glass rounded-2xl p-6 hover:scale-105 transition-all duration-300 hover:shadow-2xl relative overflow-hidden group">
      {/* Animated background gradient */}
      <div className={`absolute inset-0 bg-gradient-to-br ${gradient} opacity-0 group-hover:opacity-10 transition-opacity duration-300`}></div>

      <div className="relative z-10">
        <div className="flex items-start justify-between mb-4">
          <div className="flex-1">
            <p className="text-gray-400 text-sm font-medium mb-2">{title}</p>
            <h3 className="text-3xl font-bold text-white mb-1">{value}</h3>
            {subtitle && <p className="text-gray-500 text-xs">{subtitle}</p>}
          </div>
          {icon && (
            <div className={`p-3 rounded-xl bg-gradient-to-br ${gradient} ${pulse ? 'pulse-dot' : ''}`}>
              {icon}
            </div>
          )}
        </div>

        {trend && trendValue && (
          <div className="flex items-center gap-2">
            <span className={`text-sm font-semibold ${trendColors[trend]}`}>
              {trendIcons[trend]} {trendValue}
            </span>
            <span className="text-gray-500 text-xs">vs last period</span>
          </div>
        )}
      </div>
    </div>
  );
}
