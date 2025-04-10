// DOM Elements
const chatContainer = document.getElementById('chat-container');
const promptInput = document.getElementById('prompt-input');
const runButton = document.getElementById('run-button');
const runLoader = document.getElementById('run-loader');
const progressBar = document.getElementById('progress-bar');
const logConsole = document.getElementById('log-console');
const statusDisplay = document.getElementById('status-display');
const themeToggle = document.getElementById('theme-toggle');
const toolButtons = document.querySelectorAll('.tool-button');
const samplePrompts = document.querySelectorAll('.sample-prompt');
const sidePanel = document.getElementById('side-panel');
const closePanelBtn = document.getElementById('close-panel');

// Theme Management
function initializeTheme() {
    const savedTheme = localStorage.getItem('theme') || 'dark';
    document.body.className = `${savedTheme}-theme`;
}

function toggleTheme() {
    const isDark = document.body.classList.contains('dark-theme');
    document.body.className = isDark ? 'light-theme' : 'dark-theme';
    localStorage.setItem('theme', isDark ? 'light' : 'dark');
}

// WebSocket Communication
function initializeWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    
    console.log(`Connecting to WebSocket at ${wsUrl}`);
    
    const socket = new WebSocket(wsUrl);
    
    socket.onopen = () => {
        console.log('WebSocket connection established');
        appendLog('Connected to simulation server', 'text-success');
        enableUI();
    };
    
    socket.onmessage = (event) => {
        try {
            const message = JSON.parse(event.data);
            handleWebSocketMessage(message);
        } catch (error) {
            console.error('Error parsing message:', error);
            appendLog('Error processing server message', 'text-danger');
        }
    };
    
    socket.onclose = () => {
        console.log('WebSocket connection closed');
        appendLog('Connection to server closed. Attempting to reconnect...', 'text-warning');
        appendSystemMessage('Connection to server lost. Attempting to reconnect...');
        disableUI();
        
        // Try to reconnect after 3 seconds
        setTimeout(initializeWebSocket, 3000);
    };
    
    socket.onerror = (error) => {
        console.error('WebSocket error:', error);
        appendLog('WebSocket error: Connection failed', 'text-danger');
        appendSystemMessage('Error connecting to the simulation server.');
    };
    
    return socket;
}

// WebSocket Message Handling
function handleWebSocketMessage(message) {
    console.log('Received message:', message);
    
    switch(message.type) {
        case 'simulation_log':
            appendLog(message.log);
            updateProgressBar(message.progress || 0);
            updateStatusDisplay(`Running simulation: ${message.progress || 0}%`);
            showSidePanel();
            break;
            
        case 'simulation_complete':
            // Update UI
            hideRunLoader();
            appendLog('Simulation completed successfully!', 'text-success');
            updateStatusDisplay('Simulation completed. Analyzing results...');
            showSidePanel();
            break;
            
        case 'analysis_result':
            // Display analysis in the chat
            appendAssistantMessage(message.analysis);
            enableUI();
            updateStatusDisplay('Analysis completed.');
            break;
            
        case 'error':
            appendLog('ERROR: ' + message.message, 'text-danger');
            appendSystemMessage('Error: ' + message.message);
            hideRunLoader();
            enableUI();
            break;
            
        default:
            console.warn('Unknown message type:', message.type);
    }
}

// UI Controls
function enableUI() {
    runButton.disabled = false;
    promptInput.disabled = false;
    hideRunLoader();
    
    // Enable tool buttons and sample prompts
    toolButtons.forEach(button => button.disabled = false);
    samplePrompts.forEach(prompt => {
        prompt.style.pointerEvents = 'auto';
        prompt.style.opacity = '1';
    });
}

function disableUI() {
    runButton.disabled = true;
    promptInput.disabled = true;
    showRunLoader();
    
    // Disable tool buttons and sample prompts
    toolButtons.forEach(button => button.disabled = true);
    samplePrompts.forEach(prompt => {
        prompt.style.pointerEvents = 'none';
        prompt.style.opacity = '0.7';
    });
}

function showRunLoader() {
    runLoader.classList.remove('hidden');
}

function hideRunLoader() {
    runLoader.classList.add('hidden');
}

function updateProgressBar(percentage) {
    progressBar.style.width = `${percentage}%`;
    progressBar.textContent = `${percentage}%`;
}

function updateStatusDisplay(message) {
    statusDisplay.innerHTML = `<p>${message}</p>`;
}

function appendLog(message, className = '') {
    const logEntry = document.createElement('p');
    logEntry.textContent = message;
    if (className) {
        logEntry.className = className;
    }
    
    logConsole.appendChild(logEntry);
    logConsole.scrollTop = logConsole.scrollHeight;
}

function showSidePanel() {
    sidePanel.classList.add('open');
}

function hideSidePanel() {
    sidePanel.classList.remove('open');
}

// Chat Interface
function appendUserMessage(message) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'chat-message user-message';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = message;
    
    messageDiv.appendChild(contentDiv);
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function appendAssistantMessage(message) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'chat-message assistant-message';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = formatMessageText(message);
    
    messageDiv.appendChild(contentDiv);
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function appendSystemMessage(message) {
    const systemMessage = document.createElement('div');
    systemMessage.className = 'system-message';
    systemMessage.textContent = message;
    
    chatContainer.appendChild(systemMessage);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function formatMessageText(text) {
    // Escape HTML
    let escapedText = text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
    
    // Convert line breaks to <br>
    return escapedText.replace(/\n/g, '<br>');
}

// Auto-resize textarea
function autoResizeTextarea() {
    promptInput.style.height = 'auto';
    promptInput.style.height = (promptInput.scrollHeight) + 'px';
}

// Run Simulation
function runSimulation() {
    const prompt = promptInput.value.trim();
    
    if (prompt === '') {
        appendSystemMessage('Please enter a prompt or select a sample prompt.');
        return;
    }
    
    // Get current tool
  // Get current tool
    const activeTool = document.querySelector('.tool-button.active');
    const toolId = activeTool ? activeTool.id.replace('-btn', '') : 'chat';

    // Map tool IDs to the correct command
    let command;
    switch (toolId) {
        case 'chat':
            command = 'chat';
            break;
        case 'run-sim':
            command = 'run_simulation';
            break;
        case 'update-config':
            command = 'update_config';
            break;
        case 'analyze':
            command = 'analyze_simulation';
            break;
        default:
            command = 'chat';
    }
    
    // Add user message to chat
    appendUserMessage(prompt);
    
    // Show system message
    appendSystemMessage('Starting simulation based on your query...');
    
    // Disable UI and show loader
    disableUI();
    
    // Clear logs
    logConsole.innerHTML = '';
    appendLog('Starting simulation...', 'text-info');
    
    // Update status
    updateStatusDisplay('Starting simulation...');
    updateProgressBar(0);
    
    // Send request to server
    window.socket.send(JSON.stringify({
        command: command,
        prompt: prompt
    }));
}

// Event Handlers
function setupEventListeners() {
    // Theme toggle
    themeToggle.addEventListener('click', toggleTheme);
    
    // Run button
    runButton.addEventListener('click', runSimulation);
    
    // Enter key in textarea
    promptInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            runSimulation();
        }
    });
    
    // Textarea auto-resize
    promptInput.addEventListener('input', autoResizeTextarea);
    
    // Tool buttons
    toolButtons.forEach(button => {
        button.addEventListener('click', () => {
            // Remove active class from all buttons
            toolButtons.forEach(b => b.classList.remove('active'));
            // Add active class to clicked button
            button.classList.add('active');
        });
    });
    
    // Sample prompts
    samplePrompts.forEach(prompt => {
        prompt.addEventListener('click', () => {
            promptInput.value = prompt.textContent;
            autoResizeTextarea();
            promptInput.focus();
        });
    });
    
    // Close panel button
    closePanelBtn.addEventListener('click', hideSidePanel);
}

// Initialize
function initialize() {
    // Initialize theme
    initializeTheme();
    
    // Setup event listeners
    setupEventListeners();
    
    // Connect to WebSocket
    window.socket = initializeWebSocket();
    
    // Focus on input
    promptInput.focus();
}

// Start on page load
document.addEventListener('DOMContentLoaded', initialize);