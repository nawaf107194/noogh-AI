export interface SystemHealth {
  status: string;
  timestamp: string;
  components: {
    api: string;
    database: string;
    cache: string;
  };
}

export interface Minister {
  name: string;
  role: string;
  status: 'active' | 'idle' | 'busy';
  tasks_completed: number;
  efficiency: number;
}

export interface SystemMetrics {
  cpu_usage: number;
  memory_usage: number;
  disk_usage: number;
  active_requests: number;
  total_requests: number;
  uptime: number;
}

export interface TradingStats {
  total_trades: number;
  successful_trades: number;
  profit_loss: number;
  win_rate: number;
}
