/* 
    The main component for the chat interaface that will be used to interact with the chatbot. 
    It will display the messages between the user and the bot, and will allow the user to input a message to send to the bot. 
    The component will also display an error message if an error occurs while processing the user's request.
*/
import React, { useState, useEffect } from 'react';
import { ChevronRight } from 'lucide-react';

const ChatInterface = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isInitialLoad, setIsInitialLoad] = useState(true);

  useEffect(() => {
    fetchChatHistory();
  }, []);

  const fetchChatHistory = async () => {
    try {
      const response = await fetch('http://localhost:8000/chat_history');
      if (!response.ok) {
        throw new Error('Failed to fetch chat history');
      }
      const data = await response.json();
      if (data.history && data.history.length > 0) {
        setMessages(data.history.map(([sender, content]) => ({ type: sender === 'User' ? 'user' : 'bot', content })));
      }
    } catch (err) {
      console.error('Error fetching chat history:', err);
      // We're not setting an error state here, as we don't want to show an error for an empty history
    } finally {
      setIsInitialLoad(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: input }),
      });

      if (!response.ok) {
        throw new Error('Failed to get response');
      }

      const data = await response.json();
      setMessages(prevMessages => [...prevMessages, { type: 'user', content: input }, { type: 'bot', content: data.response }]);
      setInput('');
    } catch (err) {
      setError('An error occurred while processing your request. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full max-w-3xl mx-auto p-4">
      <div className="flex-grow overflow-auto mb-4 p-4 border rounded-lg bg-white">
        {isInitialLoad ? (
          <div className="text-center text-gray-500">Loading chat history...</div>
        ) : messages.length === 0 ? (
          <div className="text-center text-gray-500">No chat history yet. Start a conversation!</div>
        ) : (
          messages.map((message, index) => (
            <div key={index} className={`mb-2 ${message.type === 'user' ? 'text-right' : 'text-left'}`}>
              <span className={`inline-block p-2 rounded-lg ${message.type === 'user' ? 'bg-blue-100' : 'bg-gray-100'}`}>
                {message.content}
              </span>
            </div>
          ))
        )}
      </div>
      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative mb-4" role="alert">
          <strong className="font-bold">Error!</strong>
          <span className="block sm:inline"> {error}</span>
        </div>
      )}
      <form onSubmit={handleSubmit} className="flex">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          className="flex-grow p-2 border rounded-l-lg resize-none"
          placeholder="Ask a legal question..."
          disabled={isLoading}
          rows={5}
        />
        <button
          type="submit"
          className="bg-green-500 text-white p-2 rounded-r-lg hover:bg-green-600 transition-colors"
          disabled={isLoading}
        >
          {isLoading ? 'Thinking...' : <ChevronRight size={24} />}
        </button>
      </form>
    </div>
  );
};

export default ChatInterface;