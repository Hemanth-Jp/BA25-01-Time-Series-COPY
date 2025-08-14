"""
Financial Data Visualization with Plotly

This module demonstrates specialized financial chart types and analysis:
- Candlestick and OHLC charts
- Technical indicators and overlays
- Portfolio analysis and risk metrics
- Interactive financial dashboards


Version: 1.0
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def generate_financial_data():
    """
    Generate realistic financial market data for visualization examples.
    
    Returns:
        dict: Dictionary containing various financial datasets
    """
    np.random.seed(42)
    
    # Generate stock price data with realistic characteristics
    n_days = 252  # One trading year
    dates = pd.date_range('2023-01-01', periods=n_days, freq='B')  # Business days
    
    # Simulate stock price using geometric Brownian motion
    returns = np.random.normal(0.0005, 0.02, n_days)  # Daily returns
    prices = [100]  # Starting price
    
    for i in range(1, n_days):
        price = prices[i-1] * (1 + returns[i])
        prices.append(price)
    
    # Generate OHLC data
    highs = []
    lows = []
    opens = []
    closes = prices.copy()
    volumes = []
    
    for i, close in enumerate(closes):
        if i == 0:
            open_price = close
        else:
            open_price = closes[i-1] * (1 + np.random.normal(0, 0.005))
        
        daily_range = abs(np.random.normal(0, 0.015)) * close
        high = max(open_price, close) + daily_range * np.random.uniform(0, 1)
        low = min(open_price, close) - daily_range * np.random.uniform(0, 1)
        volume = np.random.randint(1000000, 5000000)
        
        opens.append(open_price)
        highs.append(high)
        lows.append(low)
        volumes.append(volume)
    
    stock_data = pd.DataFrame({
        'date': dates,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })
    
    # Portfolio data
    assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    portfolio_returns = np.random.multivariate_normal(
        mean=[0.0008, 0.0012, 0.0006, 0.0010, 0.0015],
        cov=np.array([[0.0004, 0.0002, 0.0001, 0.0001, 0.0003],
                     [0.0002, 0.0009, 0.0002, 0.0003, 0.0004],
                     [0.0001, 0.0002, 0.0003, 0.0001, 0.0002],
                     [0.0001, 0.0003, 0.0001, 0.0006, 0.0003],
                     [0.0003, 0.0004, 0.0002, 0.0003, 0.0016]]),
        size=n_days
    )
    
    portfolio_data = pd.DataFrame(portfolio_returns, columns=assets, index=dates)
    
    # Economic indicators
    economic_data = pd.DataFrame({
        'date': dates[::5],  # Weekly data
        'gdp_growth': np.random.normal(2.5, 0.5, len(dates[::5])),
        'inflation': np.random.normal(3.0, 0.8, len(dates[::5])),
        'unemployment': np.random.normal(4.2, 0.3, len(dates[::5])),
        'interest_rate': np.random.normal(2.0, 0.2, len(dates[::5]))
    })
    
    return {
        'stock': stock_data,
        'portfolio': portfolio_data,
        'economic': economic_data
    }


def create_candlestick_chart(data):
    """
    Create candlestick chart with volume and technical indicators.
    
    Args:
        data (dict): Financial data dictionary
    
    Returns:
        plotly.graph_objects.Figure: Candlestick chart figure
    """
    stock_data = data['stock']
    
    # Create subplots for price and volume
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('Stock Price', 'Volume'),
        row_width=[0.2, 0.7]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=stock_data['date'],
            open=stock_data['open'],
            high=stock_data['high'],
            low=stock_data['low'],
            close=stock_data['close'],
            name='Price',
            increasing_line_color='green',
            decreasing_line_color='red'
        ),
        row=1, col=1
    )
    
    # Calculate and add moving averages
    stock_data['ma_20'] = stock_data['close'].rolling(window=20).mean()
    stock_data['ma_50'] = stock_data['close'].rolling(window=50).mean()
    
    fig.add_trace(
        go.Scatter(
            x=stock_data['date'],
            y=stock_data['ma_20'],
            mode='lines',
            name='MA 20',
            line=dict(color='blue', width=1)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=stock_data['date'],
            y=stock_data['ma_50'],
            mode='lines',
            name='MA 50',
            line=dict(color='orange', width=1)
        ),
        row=1, col=1
    )
    
    # Volume bars
    colors = ['green' if close >= open else 'red' 
              for close, open in zip(stock_data['close'], stock_data['open'])]
    
    fig.add_trace(
        go.Bar(
            x=stock_data['date'],
            y=stock_data['volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.7
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title='Stock Price Analysis with Technical Indicators',
        template='plotly_white',
        xaxis_rangeslider_visible=False,
        height=700,
        font=dict(size=10)
    )
    
    return fig


def create_bollinger_bands(data):
    """
    Create Bollinger Bands analysis chart.
    
    Args:
        data (dict): Financial data dictionary
    
    Returns:
        plotly.graph_objects.Figure: Bollinger Bands figure
    """
    stock_data = data['stock'].copy()
    
    # Calculate Bollinger Bands
    window = 20
    stock_data['ma'] = stock_data['close'].rolling(window=window).mean()
    stock_data['std'] = stock_data['close'].rolling(window=window).std()
    stock_data['upper_band'] = stock_data['ma'] + (stock_data['std'] * 2)
    stock_data['lower_band'] = stock_data['ma'] - (stock_data['std'] * 2)
    
    fig = go.Figure()
    
    # Price line
    fig.add_trace(go.Scatter(
        x=stock_data['date'],
        y=stock_data['close'],
        mode='lines',
        name='Close Price',
        line=dict(color='black', width=2)
    ))
    
    # Moving average
    fig.add_trace(go.Scatter(
        x=stock_data['date'],
        y=stock_data['ma'],
        mode='lines',
        name='Moving Average',
        line=dict(color='blue', width=1)
    ))
    
    # Upper band
    fig.add_trace(go.Scatter(
        x=stock_data['date'],
        y=stock_data['upper_band'],
        mode='lines',
        name='Upper Band',
        line=dict(color='red', width=1, dash='dash'),
        fillcolor='rgba(255,0,0,0.1)',
        fill=None
    ))
    
    # Lower band with fill
    fig.add_trace(go.Scatter(
        x=stock_data['date'],
        y=stock_data['lower_band'],
        mode='lines',
        name='Lower Band',
        line=dict(color='red', width=1, dash='dash'),
        fillcolor='rgba(255,0,0,0.1)',
        fill='tonexty'
    ))
    
    # Identify buy/sell signals
    stock_data['position'] = 0
    stock_data.loc[stock_data['close'] < stock_data['lower_band'], 'position'] = 1  # Buy
    stock_data.loc[stock_data['close'] > stock_data['upper_band'], 'position'] = -1  # Sell
    
    buy_signals = stock_data[stock_data['position'] == 1]
    sell_signals = stock_data[stock_data['position'] == -1]
    
    # Add buy signals
    if not buy_signals.empty:
        fig.add_trace(go.Scatter(
            x=buy_signals['date'],
            y=buy_signals['close'],
            mode='markers',
            name='Buy Signal',
            marker=dict(symbol='triangle-up', size=10, color='green')
        ))
    
    # Add sell signals
    if not sell_signals.empty:
        fig.add_trace(go.Scatter(
            x=sell_signals['date'],
            y=sell_signals['close'],
            mode='markers',
            name='Sell Signal',
            marker=dict(symbol='triangle-down', size=10, color='red')
        ))
    
    fig.update_layout(
        title='Bollinger Bands Trading Strategy',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        template='plotly_white',
        font=dict(size=12),
        hovermode='x unified'
    )
    
    return fig


def create_portfolio_analysis(data):
    """
    Create portfolio performance and risk analysis.
    
    Args:
        data (dict): Financial data dictionary
    
    Returns:
        plotly.graph_objects.Figure: Portfolio analysis figure
    """
    portfolio_data = data['portfolio']
    
    # Calculate cumulative returns
    cumulative_returns = (1 + portfolio_data).cumprod()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Cumulative Returns', 'Risk-Return Scatter', 
                       'Rolling Volatility', 'Correlation Heatmap'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 1. Cumulative returns
    for asset in portfolio_data.columns:
        fig.add_trace(
            go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns[asset],
                mode='lines',
                name=asset,
                showlegend=True
            ),
            row=1, col=1
        )
    
    # 2. Risk-return scatter
    annual_returns = portfolio_data.mean() * 252
    annual_volatility = portfolio_data.std() * np.sqrt(252)
    
    fig.add_trace(
        go.Scatter(
            x=annual_volatility,
            y=annual_returns,
            mode='markers+text',
            text=portfolio_data.columns,
            textposition='top center',
            marker=dict(size=12, color='blue'),
            name='Assets',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # 3. Rolling volatility
    rolling_vol = portfolio_data.rolling(window=30).std() * np.sqrt(252)
    
    for asset in portfolio_data.columns:
        fig.add_trace(
            go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol[asset],
                mode='lines',
                name=f'{asset} Vol',
                showlegend=False
            ),
            row=2, col=1
        )
    
    # 4. Correlation heatmap
    correlation_matrix = portfolio_data.corr()
    
    fig.add_trace(
        go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.values,
            texttemplate="%{text:.2f}",
            showscale=True,
            name='Correlation',
            showlegend=False
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title='Portfolio Performance Analysis',
        template='plotly_white',
        height=800,
        font=dict(size=10)
    )
    
    # Update axis labels
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
    fig.update_xaxes(title_text="Volatility", row=1, col=2)
    fig.update_yaxes(title_text="Expected Return", row=1, col=2)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="30-Day Volatility", row=2, col=1)
    
    return fig


def create_economic_dashboard(data):
    """
    Create economic indicators dashboard.
    
    Args:
        data (dict): Financial data dictionary
    
    Returns:
        plotly.graph_objects.Figure: Economic dashboard figure
    """
    economic_data = data['economic']
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('GDP Growth vs Inflation', 'Interest Rates', 
                       'Unemployment Trend', 'Economic Indicators Summary'),
        specs=[[{"secondary_y": True}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 1. GDP Growth vs Inflation (dual axis)
    fig.add_trace(
        go.Scatter(
            x=economic_data['date'],
            y=economic_data['gdp_growth'],
            mode='lines+markers',
            name='GDP Growth (%)',
            line=dict(color='green', width=2)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=economic_data['date'],
            y=economic_data['inflation'],
            mode='lines+markers',
            name='Inflation (%)',
            line=dict(color='red', width=2),
            yaxis='y2'
        ),
        row=1, col=1, secondary_y=True
    )
    
    # 2. Interest Rates
    fig.add_trace(
        go.Scatter(
            x=economic_data['date'],
            y=economic_data['interest_rate'],
            mode='lines+markers',
            name='Interest Rate (%)',
            line=dict(color='blue', width=2),
            fill='tonexty',
            fillcolor='rgba(0,0,255,0.1)'
        ),
        row=1, col=2
    )
    
    # 3. Unemployment Trend
    fig.add_trace(
        go.Bar(
            x=economic_data['date'],
            y=economic_data['unemployment'],
            name='Unemployment (%)',
            marker_color='orange',
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # 4. Economic Indicators Summary (Box plot)
    indicators = ['GDP Growth', 'Inflation', 'Unemployment', 'Interest Rate']
    values = [economic_data['gdp_growth'], economic_data['inflation'],
              economic_data['unemployment'], economic_data['interest_rate']]
    
    for i, (indicator, value) in enumerate(zip(indicators, values)):
        fig.add_trace(
            go.Box(
                y=value,
                name=indicator,
                boxpoints='outliers',
                showlegend=False
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        title='Economic Indicators Dashboard',
        template='plotly_white',
        height=800,
        font=dict(size=10)
    )
    
    return fig


def create_options_analysis(data):
    """
    Create options pricing and Greeks analysis.
    
    Args:
        data (dict): Financial data dictionary
    
    Returns:
        plotly.graph_objects.Figure: Options analysis figure
    """
    stock_data = data['stock']
    current_price = stock_data['close'].iloc[-1]
    
    # Generate option strike prices and parameters
    strikes = np.linspace(current_price * 0.8, current_price * 1.2, 50)
    time_to_expiry = 30 / 365  # 30 days
    risk_free_rate = 0.03
    volatility = 0.25
    
    # Black-Scholes formula for call options
    def black_scholes_call(S, K, T, r, sigma):
        from scipy.stats import norm
        d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        call_price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        return call_price
    
    # Calculate option prices and Greeks
    call_prices = [black_scholes_call(current_price, k, time_to_expiry, 
                                     risk_free_rate, volatility) for k in strikes]
    
    # Delta (price sensitivity)
    delta_values = []
    for k in strikes:
        from scipy.stats import norm
        d1 = (np.log(current_price/k) + (risk_free_rate + volatility**2/2)*time_to_expiry) / (volatility*np.sqrt(time_to_expiry))
        delta = norm.cdf(d1)
        delta_values.append(delta)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Option Price vs Strike', 'Delta (Price Sensitivity)',
                       'Profit/Loss Diagram', 'Volatility Smile'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 1. Option prices
    fig.add_trace(
        go.Scatter(
            x=strikes,
            y=call_prices,
            mode='lines',
            name='Call Option Price',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # Mark current stock price
    fig.add_vline(x=current_price, line_dash="dash", line_color="red", 
                  annotation_text=f"Current Price: ${current_price:.2f}",
                  row=1, col=1)
    
    # 2. Delta
    fig.add_trace(
        go.Scatter(
            x=strikes,
            y=delta_values,
            mode='lines',
            name='Delta',
            line=dict(color='green', width=2)
        ),
        row=1, col=2
    )
    
    # 3. Profit/Loss diagram
    option_premium = black_scholes_call(current_price, current_price, 
                                       time_to_expiry, risk_free_rate, volatility)
    stock_prices_at_expiry = strikes
    payoffs = np.maximum(stock_prices_at_expiry - current_price, 0) - option_premium
    
    fig.add_trace(
        go.Scatter(
            x=stock_prices_at_expiry,
            y=payoffs,
            mode='lines',
            name='P&L',
            line=dict(color='purple', width=2),
            fill='tonexty',
            fillcolor='rgba(128,0,128,0.1)'
        ),
        row=2, col=1
    )
    
    # Add break-even line
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=2, col=1)
    
    # 4. Volatility smile (simulated)
    vol_strikes = np.linspace(current_price * 0.9, current_price * 1.1, 20)
    implied_vols = [volatility + 0.05 * abs(k/current_price - 1)**2 for k in vol_strikes]
    
    fig.add_trace(
        go.Scatter(
            x=vol_strikes/current_price,  # Moneyness
            y=implied_vols,
            mode='markers+lines',
            name='Implied Volatility',
            marker=dict(size=8, color='orange'),
            line=dict(color='orange', width=2)
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title='Options Analysis Dashboard',
        template='plotly_white',
        height=800,
        font=dict(size=10)
    )
    
    # Update axis labels
    fig.update_xaxes(title_text="Strike Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Option Price ($)", row=1, col=1)
    fig.update_xaxes(title_text="Strike Price ($)", row=1, col=2)
    fig.update_yaxes(title_text="Delta", row=1, col=2)
    fig.update_xaxes(title_text="Stock Price at Expiry ($)", row=2, col=1)
    fig.update_yaxes(title_text="Profit/Loss ($)", row=2, col=1)
    fig.update_xaxes(title_text="Moneyness (S/K)", row=2, col=2)
    fig.update_yaxes(title_text="Implied Volatility", row=2, col=2)
    
    return fig


def main():
    """
    Main function demonstrating financial visualization capabilities.
    
    Creates comprehensive financial analysis charts including candlestick charts,
    technical indicators, portfolio analysis, and options pricing.
    """
    print("=== Financial Data Visualization Examples ===\n")
    
    try:
        # Generate financial data
        print("Generating financial market data...")
        data = generate_financial_data()
        print("Financial data generated successfully\n")
        
        # Create candlestick chart
        print("Creating candlestick chart with technical indicators...")
        candlestick_fig = create_candlestick_chart(data)
        candlestick_fig.show()
        print("Candlestick chart created\n")
        
        # Create Bollinger Bands analysis
        print("Creating Bollinger Bands analysis...")
        bollinger_fig = create_bollinger_bands(data)
        bollinger_fig.show()
        print("Bollinger Bands analysis created\n")
        
        # Create portfolio analysis
        print("Creating portfolio performance analysis...")
        portfolio_fig = create_portfolio_analysis(data)
        portfolio_fig.show()
        print("Portfolio analysis created\n")
        
        # Create economic dashboard
        print("Creating economic indicators dashboard...")
        economic_fig = create_economic_dashboard(data)
        economic_fig.show()
        print("Economic dashboard created\n")
        
        # Create options analysis
        print("Creating options analysis...")
        options_fig = create_options_analysis(data)
        options_fig.show()
        print("Options analysis created\n")
        
        print("=== All financial visualizations completed successfully ===")
        
        return {
            'candlestick': candlestick_fig,
            'bollinger': bollinger_fig,
            'portfolio': portfolio_fig,
            'economic': economic_fig,
            'options': options_fig
        }
        
    except Exception as e:
        print(f"Error in financial visualization creation: {str(e)}")
        return None


if __name__ == "__main__":
    results = main()