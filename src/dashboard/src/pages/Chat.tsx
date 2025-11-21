import { useState, useEffect, useRef } from 'react';
import { Send, Bot, User, Loader2, Sparkles } from 'lucide-react';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

export function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  // Load chat history from localStorage
  useEffect(() => {
    const saved = localStorage.getItem('noogh_chat_history');
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        setMessages(parsed.map((msg: any) => ({
          ...msg,
          timestamp: new Date(msg.timestamp)
        })));
      } catch (e) {
        console.error('Failed to load chat history:', e);
      }
    }
  }, []);

  // Save chat history to localStorage
  useEffect(() => {
    if (messages.length > 0) {
      localStorage.setItem('noogh_chat_history', JSON.stringify(messages));
    }
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input.trim(),
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: userMessage.content,
          history: messages.slice(-5).map(m => ({
            role: m.role,
            content: m.content
          }))
        })
      });

      if (!response.ok) {
        throw new Error('Failed to get response from Noogh');
      }

      const data = await response.json();

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.response || data.message || 'عذراً، لم أستطع فهم سؤالك.',
        timestamp: new Date()
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Chat error:', error);

      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'عذراً، حدث خطأ في الاتصال. تأكد من أن الخادم يعمل.',
        timestamp: new Date()
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const clearChat = () => {
    if (confirm('هل تريد مسح تاريخ المحادثة؟')) {
      setMessages([]);
      localStorage.removeItem('noogh_chat_history');
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      {/* Header */}
      <div className="glass border-b border-gray-700 p-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="relative">
              <div className="w-12 h-12 rounded-full bg-gradient-to-r from-blue-500 to-purple-600 flex items-center justify-center">
                <Bot size={28} className="text-white" />
              </div>
              <div className="absolute -bottom-1 -right-1 w-4 h-4 bg-green-500 rounded-full border-2 border-slate-900"></div>
            </div>
            <div>
              <h1 className="text-2xl font-bold gradient-text flex items-center gap-2">
                Chat with Noogh
                <Sparkles size={20} className="text-yellow-400" />
              </h1>
              <p className="text-gray-400 text-sm">نظام الذكاء الاصطناعي الموحد</p>
            </div>
          </div>
          <button
            onClick={clearChat}
            className="px-4 py-2 rounded-lg bg-red-500/10 text-red-400 hover:bg-red-500/20 transition-colors text-sm"
          >
            مسح المحادثة
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <div className="w-20 h-20 rounded-full bg-gradient-to-r from-blue-500 to-purple-600 flex items-center justify-center mb-4">
              <Bot size={40} className="text-white" />
            </div>
            <h2 className="text-2xl font-bold text-white mb-2">مرحباً! أنا نوغ</h2>
            <p className="text-gray-400 max-w-md">
              أنا نظام ذكاء اصطناعي موحد. اسألني عن أي شيء متعلق بالنظام، التدريب، المعرفة، أو أي موضوع آخر!
            </p>
            <div className="grid grid-cols-2 gap-3 mt-8 max-w-2xl">
              {[
                'ما هي حالة النظام؟',
                'كم قطعة معرفية لديك؟',
                'اشرح لي Brain v4.0',
                'ما هو MCP Server؟'
              ].map((suggestion, i) => (
                <button
                  key={i}
                  onClick={() => setInput(suggestion)}
                  className="glass px-4 py-3 rounded-xl text-sm text-gray-300 hover:text-white hover:border-blue-500/50 border border-gray-700 transition-all"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex gap-4 ${message.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}
          >
            <div className={`flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center ${
              message.role === 'user'
                ? 'bg-gradient-to-r from-green-500 to-emerald-600'
                : 'bg-gradient-to-r from-blue-500 to-purple-600'
            }`}>
              {message.role === 'user' ? <User size={20} className="text-white" /> : <Bot size={20} className="text-white" />}
            </div>

            <div className={`flex-1 max-w-3xl ${message.role === 'user' ? 'items-end' : 'items-start'} flex flex-col gap-2`}>
              <div className={`glass rounded-2xl p-4 ${
                message.role === 'user'
                  ? 'bg-gradient-to-r from-green-500/10 to-emerald-600/10 border-green-500/20'
                  : 'bg-gradient-to-r from-blue-500/10 to-purple-600/10 border-blue-500/20'
              } border`}>
                <p className="text-white leading-relaxed whitespace-pre-wrap">{message.content}</p>
              </div>
              <span className="text-xs text-gray-500">
                {message.timestamp.toLocaleTimeString('ar-SA', { hour: '2-digit', minute: '2-digit' })}
              </span>
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="flex gap-4">
            <div className="flex-shrink-0 w-10 h-10 rounded-full bg-gradient-to-r from-blue-500 to-purple-600 flex items-center justify-center">
              <Bot size={20} className="text-white" />
            </div>
            <div className="glass rounded-2xl p-4 bg-gradient-to-r from-blue-500/10 to-purple-600/10 border border-blue-500/20">
              <div className="flex items-center gap-2 text-gray-400">
                <Loader2 size={16} className="animate-spin" />
                <span>نوغ يفكر...</span>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="glass border-t border-gray-700 p-6">
        <div className="flex gap-4 max-w-4xl mx-auto">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="اكتب رسالتك هنا... (Enter للإرسال)"
            disabled={isLoading}
            className="flex-1 px-6 py-4 rounded-xl bg-slate-800 border border-gray-700 text-white placeholder-gray-500 focus:outline-none focus:border-blue-500 transition-colors disabled:opacity-50"
          />
          <button
            onClick={handleSend}
            disabled={!input.trim() || isLoading}
            className="px-6 py-4 rounded-xl bg-gradient-to-r from-blue-500 to-purple-600 text-white font-bold hover:shadow-lg hover:scale-105 transition-all disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 flex items-center gap-2"
          >
            {isLoading ? (
              <Loader2 size={20} className="animate-spin" />
            ) : (
              <>
                <Send size={20} />
                <span className="hidden sm:inline">إرسال</span>
              </>
            )}
          </button>
        </div>
        <p className="text-center text-xs text-gray-500 mt-3">
          نوغ هو نظام ذكاء اصطناعي. قد يُخطئ أحياناً. تحقق من المعلومات المهمة.
        </p>
      </div>
    </div>
  );
}
