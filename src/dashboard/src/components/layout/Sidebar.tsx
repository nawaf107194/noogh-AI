import { Link, useLocation } from 'react-router-dom';
import { Home, Users, TrendingUp, Settings, FileText, Moon, Sun, Bot, Zap } from 'lucide-react';
import { useTheme } from '../../contexts/ThemeContext';

const navItems = [
  { path: '/', icon: Home, label: 'Dashboard' },
  { path: '/ministers', icon: Users, label: 'Ministers' },
  { path: '/automation', icon: Zap, label: 'Automation' },
  { path: '/chat', icon: Bot, label: 'Chat' },
  { path: '/trading', icon: TrendingUp, label: 'Trading' },
  { path: '/reports', icon: FileText, label: 'Reports' },
  { path: '/settings', icon: Settings, label: 'Settings' },
];

export function Sidebar() {
  const location = useLocation();
  const { theme, toggleTheme } = useTheme();

  return (
    <aside className="w-64 glass border-r border-gray-700 min-h-screen sticky top-0 flex flex-col">
      <div className="p-6 border-b border-gray-700">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center animate-gradient">
            <span className="text-white font-bold text-xl">نـ</span>
          </div>
          <div>
            <h1 className="text-lg font-bold text-white">Noogh</h1>
            <p className="text-xs text-gray-400">Unified System</p>
          </div>
        </div>
      </div>

      <nav className="flex-1 p-4">
        <ul className="space-y-2">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = location.pathname === item.path;

            return (
              <li key={item.path}>
                <Link
                  to={item.path}
                  className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200 ${
                    isActive
                      ? 'bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-lg'
                      : 'text-gray-400 hover:bg-gray-800 hover:text-white'
                  }`}
                >
                  <Icon size={20} />
                  <span className="font-medium">{item.label}</span>
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>

      <div className="p-4 border-t border-gray-700">
        <button
          onClick={toggleTheme}
          className="w-full flex items-center gap-3 px-4 py-3 rounded-lg text-gray-400 hover:bg-gray-800 hover:text-white transition-all duration-200"
        >
          {theme === 'dark' ? <Sun size={20} /> : <Moon size={20} />}
          <span className="font-medium">
            {theme === 'dark' ? 'Light Mode' : 'Dark Mode'}
          </span>
        </button>
      </div>
    </aside>
  );
}
