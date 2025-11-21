import { Settings as SettingsIcon, Bell, Shield, Zap } from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';

export function Settings() {
  const { theme, toggleTheme } = useTheme();

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold gradient-text mb-2">Settings</h1>
        <p className="text-gray-400">Configure your Noogh system preferences</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Appearance */}
        <div className="glass rounded-2xl p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 rounded-lg bg-gradient-to-br from-purple-500 to-pink-600">
              <SettingsIcon className="text-white" size={20} />
            </div>
            <h3 className="text-lg font-bold text-white">Appearance</h3>
          </div>
          <div className="space-y-4">
            <div className="flex items-center justify-between p-4 bg-slate-800 bg-opacity-50 rounded-lg">
              <div>
                <p className="text-white font-medium">Theme</p>
                <p className="text-gray-400 text-sm">Choose your preferred theme</p>
              </div>
              <button
                onClick={toggleTheme}
                className="px-4 py-2 rounded-lg bg-gradient-to-r from-blue-500 to-purple-600 text-white text-sm hover:shadow-lg transition-all"
              >
                {theme === 'dark' ? 'Dark' : 'Light'}
              </button>
            </div>
          </div>
        </div>

        {/* Notifications */}
        <div className="glass rounded-2xl p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 rounded-lg bg-gradient-to-br from-blue-500 to-cyan-600">
              <Bell className="text-white" size={20} />
            </div>
            <h3 className="text-lg font-bold text-white">Notifications</h3>
          </div>
          <div className="space-y-4">
            {['System Alerts', 'Trading Updates', 'Minister Activity'].map((item, i) => (
              <div key={i} className="flex items-center justify-between p-4 bg-slate-800 bg-opacity-50 rounded-lg">
                <p className="text-white">{item}</p>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input type="checkbox" className="sr-only peer" defaultChecked />
                  <div className="w-11 h-6 bg-gray-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-gradient-to-r peer-checked:from-blue-500 peer-checked:to-purple-600"></div>
                </label>
              </div>
            ))}
          </div>
        </div>

        {/* Security */}
        <div className="glass rounded-2xl p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 rounded-lg bg-gradient-to-br from-red-500 to-orange-600">
              <Shield className="text-white" size={20} />
            </div>
            <h3 className="text-lg font-bold text-white">Security</h3>
          </div>
          <div className="space-y-3">
            <div className="p-4 bg-slate-800 bg-opacity-50 rounded-lg">
              <p className="text-white font-medium mb-1">API Key</p>
              <p className="text-gray-400 text-sm font-mono">dev-test-key-***************</p>
            </div>
            <button className="w-full px-4 py-2 rounded-lg bg-gradient-to-r from-red-500 to-orange-600 text-white hover:shadow-lg transition-all">
              Regenerate API Key
            </button>
          </div>
        </div>

        {/* System */}
        <div className="glass rounded-2xl p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 rounded-lg bg-gradient-to-br from-green-500 to-emerald-600">
              <Zap className="text-white" size={20} />
            </div>
            <h3 className="text-lg font-bold text-white">System</h3>
          </div>
          <div className="space-y-3">
            <div className="p-4 bg-slate-800 bg-opacity-50 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="text-gray-400 text-sm">Version</span>
                <span className="text-white font-medium">v1.0.0</span>
              </div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-gray-400 text-sm">Uptime</span>
                <span className="text-white font-medium">15d 8h 42m</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-400 text-sm">Status</span>
                <span className="flex items-center gap-2 text-green-400 font-medium">
                  <span className="w-2 h-2 bg-green-500 rounded-full pulse-dot"></span>
                  Online
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
