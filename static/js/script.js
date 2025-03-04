// DOM Elements
const productSelect = document.getElementById('product');
const modelSelect = document.getElementById('model');
const granularitySelect = document.getElementById('granularity');
const marginSelect = document.getElementById('margin');
const marginAmountInput = document.getElementById('margin-amount');
const leverageSlider = document.getElementById('leverage');
const leverageValue = document.getElementById('leverage-value');
const tpSlSlider = document.getElementById('tp-sl');
const tpSlValue = document.getElementById('tp-sl-value');
const limitOrderCheckbox = document.getElementById('limit-order');
const riskLevelSelect = document.getElementById('risk-level');
const tradingOptionsToggle = document.getElementById('trading-options-toggle');
const tradingOptionsPanel = document.querySelector('.trading-options');
const analyzeBtn = document.getElementById('analyze-btn');
const longBtn = document.getElementById('long-btn');
const shortBtn = document.getElementById('short-btn');
const closePositionsBtn = document.getElementById('close-positions-btn');
const checkOrdersBtn = document.getElementById('check-orders-btn');
const autoTradeBtn = document.getElementById('auto-trade-btn');
const clearBtn = document.getElementById('clear-btn');
const cancelBtn = document.getElementById('cancel-btn');
const outputConsole = document.getElementById('output-console');
const statusIndicator = document.getElementById('status-indicator');
const currentPrice = document.getElementById('current-price');
const priceTime = document.getElementById('price-time');
const winCount = document.getElementById('win-count');
const lossCount = document.getElementById('loss-count');
const winRate = document.getElementById('win-rate');
const tradeHistoryList = document.getElementById('trade-history-list');

// Global variables
let autoTrading = false;
let tradeChart = null;
let eventSource = null;
let config = {
    product: 'BTC-USDC',
    model: 'o1_mini',
    granularity: 'ONE_HOUR',
    margin: 'CROSS',
    margin_amount: 60,
    leverage: 10,
    take_profit: 2.0,
    stop_loss: 2.0,
    limit_order: false,
    risk_level: 'medium'
};

// Initialize the application
function init() {
    // Set up event listeners
    setupEventListeners();
    
    // Load configuration
    loadConfig();
    
    // Load trade history
    loadTradeHistory();
    
    // Initialize trade chart
    initTradeChart();
    
    // Fetch current price immediately
    fetchCurrentPrice();
    
    // Connect to server events
    connectToEventSource();
}

// Set up event listeners
function setupEventListeners() {
    // Configuration changes
    productSelect.addEventListener('change', updateConfig);
    modelSelect.addEventListener('change', updateConfig);
    granularitySelect.addEventListener('change', updateConfig);
    marginSelect.addEventListener('change', updateConfig);
    marginAmountInput.addEventListener('change', updateConfig);
    leverageSlider.addEventListener('input', updateLeverageLabel);
    leverageSlider.addEventListener('change', updateConfig);
    tpSlSlider.addEventListener('input', updateTpSlLabel);
    tpSlSlider.addEventListener('change', updateConfig);
    limitOrderCheckbox.addEventListener('change', updateConfig);
    riskLevelSelect.addEventListener('change', updateConfig);
    
    // Price display click to refresh
    document.querySelector('.price-display').addEventListener('click', fetchCurrentPrice);
    
    // Trading options toggle
    tradingOptionsToggle.addEventListener('change', toggleTradingOptions);
    
    // Make the entire card header clickable
    document.querySelector('.trading-options-card .card-header').addEventListener('click', function(e) {
        // Don't toggle if clicking directly on the checkbox (it will handle its own state)
        if (e.target !== tradingOptionsToggle) {
            tradingOptionsToggle.checked = !tradingOptionsToggle.checked;
            toggleTradingOptions();
        }
    });
    
    // Action buttons
    analyzeBtn.addEventListener('click', runAnalysis);
    longBtn.addEventListener('click', () => placeOrder('LONG'));
    shortBtn.addEventListener('click', () => placeOrder('SHORT'));
    closePositionsBtn.addEventListener('click', closePositions);
    checkOrdersBtn.addEventListener('click', checkOpenOrders);
    autoTradeBtn.addEventListener('click', toggleAutoTrading);
    clearBtn.addEventListener('click', clearOutput);
    cancelBtn.addEventListener('click', cancelOperation);
}

// Load configuration from server
function loadConfig() {
    fetch('/api/config')
        .then(response => response.json())
        .then(data => {
            config = data;
            updateConfigUI();
        })
        .catch(error => console.error('Error loading config:', error));
}

// Update UI based on config
function updateConfigUI() {
    productSelect.value = config.product;
    modelSelect.value = config.model;
    granularitySelect.value = config.granularity;
    marginSelect.value = config.margin;
    marginAmountInput.value = config.margin_amount;
    leverageSlider.value = config.leverage;
    leverageValue.textContent = config.leverage;
    tpSlSlider.value = config.take_profit;
    tpSlValue.textContent = config.take_profit;
    limitOrderCheckbox.checked = config.limit_order;
    riskLevelSelect.value = config.risk_level;
}

// Update configuration on server
function updateConfig() {
    // Update local config
    config.product = productSelect.value;
    config.model = modelSelect.value;
    config.granularity = granularitySelect.value;
    config.margin = marginSelect.value;
    config.margin_amount = parseInt(marginAmountInput.value) || 60; // Default to 60 if invalid
    config.leverage = parseInt(leverageSlider.value);
    config.take_profit = parseFloat(tpSlSlider.value);
    config.stop_loss = parseFloat(tpSlSlider.value);
    config.limit_order = limitOrderCheckbox.checked;
    config.risk_level = riskLevelSelect.value;
    
    // Send to server
    fetch('/api/config', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(config)
    })
    .then(response => response.json())
    .then(data => {
        console.log('Config updated:', data);
        
        // If product changed, fetch new price
        if (data.config && data.config.product !== config.product) {
            fetchCurrentPrice();
        }
    })
    .catch(error => console.error('Error updating config:', error));
    
    // If product changed, fetch new price immediately
    // This is needed because the server response might not reflect the product change
    if (config.product !== productSelect.value) {
        fetchCurrentPrice();
    }
}

// Update leverage label
function updateLeverageLabel() {
    leverageValue.textContent = leverageSlider.value;
}

// Update TP/SL label
function updateTpSlLabel() {
    tpSlValue.textContent = tpSlSlider.value;
}

// Toggle trading options panel
function toggleTradingOptions() {
    if (tradingOptionsToggle.checked) {
        tradingOptionsPanel.style.display = 'block';
    } else {
        tradingOptionsPanel.style.display = 'none';
    }
}

// Run market analysis
function runAnalysis() {
    // Disable analyze button
    analyzeBtn.disabled = true;
    
    // Update status
    updateStatus('Running analysis...', 'bg-primary');
    
    // Clear output
    // outputConsole.innerHTML = '';
    
    // Send analysis request
    fetch('/api/analysis', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            granularity: config.granularity
        })
    })
    .then(response => response.json())
    .then(data => {
        console.log('Analysis started:', data);
    })
    .catch(error => {
        console.error('Error starting analysis:', error);
        updateStatus('Error', 'bg-danger');
        analyzeBtn.disabled = false;
    });
}

// Place market order
function placeOrder(side) {
    // Disable order buttons
    longBtn.disabled = true;
    shortBtn.disabled = true;
    
    // Update status
    updateStatus(`Placing ${side} order...`, 'bg-warning');
    
    // Send order request
    fetch('/api/order', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            side: side
        })
    })
    .then(response => response.json())
    .then(data => {
        console.log('Order initiated:', data);
    })
    .catch(error => {
        console.error('Error placing order:', error);
        updateStatus('Error', 'bg-danger');
        longBtn.disabled = false;
        shortBtn.disabled = false;
    });
}

// Close all positions
function closePositions() {
    // Disable button
    closePositionsBtn.disabled = true;
    
    // Update status
    updateStatus('Closing positions...', 'bg-warning');
    
    // Send close positions request
    fetch('/api/close-positions', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        console.log('Closing positions:', data);
    })
    .catch(error => {
        console.error('Error closing positions:', error);
        updateStatus('Error', 'bg-danger');
        closePositionsBtn.disabled = false;
    });
}

// Check open orders
function checkOpenOrders() {
    // Add message to console
    appendToConsole('Checking for open orders and positions...');
    
    // Send check orders request
    fetch('/api/check-orders', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        console.log('Checking orders:', data);
    })
    .catch(error => {
        console.error('Error checking orders:', error);
        appendToConsole('Error checking orders: ' + error);
    });
}

// Toggle auto-trading
function toggleAutoTrading() {
    // Send auto-trading request
    fetch('/api/auto-trading', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        console.log('Auto-trading toggled:', data);
        autoTrading = data.auto_trading;
        
        // Update button text and style
        if (autoTrading) {
            autoTradeBtn.textContent = 'Stop Auto-Trading';
            autoTradeBtn.classList.remove('btn-indigo');
            autoTradeBtn.classList.add('btn-danger');
            updateStatus('Auto-trading ON', 'bg-success');
        } else {
            autoTradeBtn.textContent = 'Start Auto-Trading';
            autoTradeBtn.classList.remove('btn-danger');
            autoTradeBtn.classList.add('btn-indigo');
            updateStatus('Auto-trading OFF', 'bg-secondary');
        }
    })
    .catch(error => {
        console.error('Error toggling auto-trading:', error);
        updateStatus('Error', 'bg-danger');
    });
}

// Clear output console
function clearOutput() {
    outputConsole.innerHTML = '';
}

// Cancel current operation
function cancelOperation() {
    // Send cancel request
    fetch('/api/cancel', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        console.log('Operation cancelled:', data);
        updateStatus('Cancelled', 'bg-secondary');
        
        // Re-enable buttons
        analyzeBtn.disabled = false;
        longBtn.disabled = false;
        shortBtn.disabled = false;
        closePositionsBtn.disabled = false;
    })
    .catch(error => {
        console.error('Error cancelling operation:', error);
    });
}

// Update status indicator
function updateStatus(text, className) {
    statusIndicator.textContent = text;
    
    // Remove all background classes
    statusIndicator.className = '';
    
    // Add new class
    statusIndicator.classList.add(className);
}

// Append text to console
function appendToConsole(text) {
    // Create new div for the message
    const messageDiv = document.createElement('div');
    
    // Escape HTML entities to prevent XSS but preserve formatting
    const escapedText = text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
    
    // Set inner HTML instead of text content to preserve formatting
    messageDiv.innerHTML = escapedText;
    
    // Add to console
    outputConsole.appendChild(messageDiv);
    
    // Scroll to bottom
    outputConsole.scrollTop = outputConsole.scrollHeight;
}

// Connect to server-sent events
function connectToEventSource() {
    // Close existing connection if any
    if (eventSource) {
        eventSource.close();
    }
    
    // Create new event source
    eventSource = new EventSource('/api/events');
    
    // Handle messages
    eventSource.onmessage = function(event) {
        const data = JSON.parse(event.data);
        
        // Handle different message types
        switch (data.type) {
            case 'message':
                appendToConsole(data.message);
                break;
                
            case 'price_update':
                updatePrice(data.price, data.timestamp);
                // If there's a note, append it to the console
                if (data.note) {
                    appendToConsole('Note: ' + data.note);
                }
                break;
                
            case 'status':
                handleStatusUpdate(data.status);
                break;
                
            case 'error':
                appendToConsole('ERROR: ' + data.message);
                updateStatus('Error', 'bg-danger');
                break;
                
            case 'ping':
                // Ignore ping messages
                break;
                
            default:
                console.log('Unknown message type:', data);
        }
    };
    
    // Handle errors
    eventSource.onerror = function(error) {
        console.error('EventSource error:', error);
        
        // Try to reconnect after a delay
        setTimeout(connectToEventSource, 5000);
    };
}

// Update price display
function updatePrice(price, timestamp) {
    // Check if price is valid
    if (price && !isNaN(parseFloat(price)) && parseFloat(price) > 0) {
        currentPrice.textContent = parseFloat(price).toFixed(2);
        currentPrice.style.color = '#4caf50'; // Green color for valid price
    } else {
        currentPrice.textContent = 'Unavailable';
        currentPrice.style.color = '#f44336'; // Red color for invalid price
    }
    
    priceTime.textContent = 'Last update: ' + timestamp;
    
    // Log price update for debugging
    console.log('Price update:', { price, timestamp });
}

// Handle status updates
function handleStatusUpdate(status) {
    // Update status indicator
    updateStatus(status, 'bg-info');
    
    // Re-enable buttons based on status
    if (status === 'Analysis completed' || status === 'Analysis failed') {
        analyzeBtn.disabled = false;
    } else if (status === 'Order completed' || status === 'Order failed') {
        longBtn.disabled = false;
        shortBtn.disabled = false;
    } else if (status.includes('positions')) {
        closePositionsBtn.disabled = false;
    }
}

// Load trade history
function loadTradeHistory() {
    fetch('/api/trade-history')
        .then(response => response.json())
        .then(data => {
            updateTradeHistory(data);
        })
        .catch(error => console.error('Error loading trade history:', error));
}

// Update trade history display
function updateTradeHistory(history) {
    // Clear existing history
    tradeHistoryList.innerHTML = '';
    
    if (history.length === 0) {
        // Show no trades message
        const noTradesDiv = document.createElement('div');
        noTradesDiv.className = 'text-center text-muted';
        noTradesDiv.textContent = 'No trades yet';
        tradeHistoryList.appendChild(noTradesDiv);
        return;
    }
    
    // Count wins and losses
    let wins = 0;
    let losses = 0;
    
    // Process trade history (most recent first)
    history.slice().reverse().forEach(trade => {
        // Count wins and losses
        if (trade.result === 'success') {
            wins++;
        } else if (trade.result === 'failure') {
            losses++;
        }
        
        // Create trade item
        const tradeItem = document.createElement('div');
        tradeItem.className = 'trade-item';
        
        // Add result class
        if (trade.result === 'success') {
            tradeItem.classList.add('trade-success');
        } else if (trade.result === 'failure') {
            tradeItem.classList.add('trade-failure');
        }
        
        // Set content
        tradeItem.textContent = `${trade.timestamp} - ${trade.product} (${trade.granularity.replace('_', ' ').toLowerCase()}) - ${trade.result}`;
        
        // Add to list
        tradeHistoryList.appendChild(tradeItem);
    });
    
    // Update stats
    winCount.textContent = wins;
    lossCount.textContent = losses;
    
    // Calculate win rate
    const total = wins + losses;
    const winRateValue = total > 0 ? Math.round((wins / total) * 100) : 0;
    winRate.textContent = `${winRateValue}%`;
    
    // Update chart
    updateTradeChart(wins, losses);
}

// Initialize trade chart
function initTradeChart() {
    const ctx = document.getElementById('trade-chart').getContext('2d');
    
    tradeChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Wins', 'Losses'],
            datasets: [{
                data: [0, 0],
                backgroundColor: ['#4caf50', '#f44336'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: '#e0e0e0'
                    }
                }
            }
        }
    });
}

// Update trade chart
function updateTradeChart(wins, losses) {
    if (tradeChart) {
        tradeChart.data.datasets[0].data = [wins, losses];
        tradeChart.update();
    }
}

// Fetch current price
function fetchCurrentPrice() {
    // Show loading indicator
    currentPrice.textContent = 'Loading...';
    currentPrice.style.color = '#999';
    
    fetch('/api/current-price')
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.error('Error fetching price:', data.error);
                updatePrice(null, 'Failed to fetch');
            } else {
                updatePrice(data.price, data.timestamp);
                if (data.note) {
                    appendToConsole('Note: ' + data.note);
                }
            }
        })
        .catch(error => {
            console.error('Error fetching price:', error);
            updatePrice(null, 'Failed to fetch');
        });
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', init); 