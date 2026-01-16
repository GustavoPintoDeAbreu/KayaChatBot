import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Send, LogIn, LogOut, MessageCircle, Database } from 'lucide-react';
import './App.css';

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [email, setEmail] = useState('');
  const [token, setToken] = useState('');
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [conversationId, setConversationId] = useState(null);
  const [ragUsed, setRagUsed] = useState(false);
  const [chunksRetrieved, setChunksRetrieved] = useState(0);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Check if user is already logged in
    const savedToken = localStorage.getItem('authToken');
    const savedEmail = localStorage.getItem('userEmail');
    if (savedToken && savedEmail) {
      setToken(savedToken);
      setEmail(savedEmail);
      setIsAuthenticated(true);
    }
  }, []);

  const handleLogin = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post('/api/auth/login', { email });
      setToken(response.data.token);
      setIsAuthenticated(true);
      localStorage.setItem('authToken', response.data.token);
      localStorage.setItem('userEmail', email);
    } catch (error) {
      alert('Login failed: ' + (error.response?.data?.detail || 'Unknown error'));
    }
  };

  const handleLogout = () => {
    setIsAuthenticated(false);
    setToken('');
    setEmail('');
    setMessages([]);
    setConversationId(null);
    localStorage.removeItem('authToken');
    localStorage.removeItem('userEmail');
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!inputMessage.trim()) return;

    const userMessage = { role: 'user', content: inputMessage };
    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const response = await axios.post('/api/chat', {
        message: inputMessage,
        conversation_id: conversationId
      }, {
        headers: { Authorization: `Bearer ${token}` }
      });

      const botMessage = { role: 'assistant', content: response.data.response };
      setMessages(prev => [...prev, botMessage]);
      setConversationId(response.data.conversation_id);
      setRagUsed(response.data.rag_context_used);
      setChunksRetrieved(response.data.retrieved_chunks);

    } catch (error) {
      const errorMessage = {
        role: 'assistant',
        content: 'Erro: ' + (error.response?.data?.detail || 'Falha na comunicação'),
        isError: true
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  if (!isAuthenticated) {
    return (
      <div className="login-container">
        <div className="login-card">
          <div className="login-header">
            <MessageCircle size={48} className="login-icon" />
            <h1>Kaya Chatbot</h1>
            <p>Entre com seu email para conversar</p>
          </div>
          <form onSubmit={handleLogin} className="login-form">
            <input
              type="email"
              placeholder="seu@email.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              className="email-input"
            />
            <button type="submit" className="login-button">
              <LogIn size={20} />
              Entrar
            </button>
          </form>
        </div>
      </div>
    );
  }

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <div className="header-left">
            <MessageCircle size={24} />
            <h1>Kaya Chatbot</h1>
          </div>
          <div className="header-right">
            <span className="user-email">{email}</span>
            <button onClick={handleLogout} className="logout-button">
              <LogOut size={16} />
              Sair
            </button>
          </div>
        </div>
      </header>

      <div className="chat-container">
        <div className="messages-container">
          {messages.length === 0 && (
            <div className="welcome-message">
              <MessageCircle size={48} className="welcome-icon" />
              <h2>Olá! 👋</h2>
              <p>Eu sou a Kaya, pronta para conversar sobre o nosso grupo do WhatsApp.</p>
              <p>Faça perguntas sobre conversas passadas ou apenas converse casualmente!</p>
            </div>
          )}

          {messages.map((message, index) => (
            <div
              key={index}
              className={`message ${message.role === 'user' ? 'user-message' : 'bot-message'} ${message.isError ? 'error-message' : ''}`}
            >
              <div className="message-content">
                {message.content}
              </div>
              {message.role === 'assistant' && index === messages.length - 1 && ragUsed && (
                <div className="rag-indicator">
                  <Database size={14} />
                  Usou {chunksRetrieved} conversas do histórico
                </div>
              )}
            </div>
          ))}

          {isLoading && (
            <div className="message bot-message loading">
              <div className="message-content">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        <form onSubmit={handleSendMessage} className="input-form">
          <input
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            placeholder="Digite sua mensagem..."
            disabled={isLoading}
            className="message-input"
          />
          <button
            type="submit"
            disabled={isLoading || !inputMessage.trim()}
            className="send-button"
          >
            <Send size={20} />
          </button>
        </form>
      </div>
    </div>
  );
}

export default App;