//The main component that composes the entire App.
import React from 'react';
import Header from './components/Header';
import ChatInterface from './components/ChatInterface';

function App() {
  return (
    <div className="flex flex-col min-h-screen">
      <Header />
      <main className="flex-grow">
        <ChatInterface />
      </main>
    </div>
  );
}

export default App;