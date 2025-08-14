"""
Interactive and Animated Plots with Matplotlib

This module demonstrates interactive widgets and animations:
- Interactive parameter adjustment
- Real-time data updates
- Animation techniques
- Event handling


Version: 1.0
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
import numpy as np

def create_interactive_plot():
    """
    Create interactive plot with parameter sliders.
    
    Demonstrates:
    - Interactive widgets
    - Real-time plot updates
    - Parameter adjustment
    - Event handling
    """
    # Initial parameters
    freq_init = 1.0
    amp_init = 1.0
    
    # Generate data
    t = np.linspace(0, 10, 1000)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.25)
    
    # Initial plot
    line, = ax.plot(t, amp_init * np.sin(2 * np.pi * freq_init * t), 
                    'b-', linewidth=2, label='sin(2*pi*f*t)')
    ax.set_xlim(0, 10)
    ax.set_ylim(-3, 3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Interactive Sine Wave', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Create slider axes
    ax_freq = plt.axes([0.2, 0.1, 0.5, 0.03])
    ax_amp = plt.axes([0.2, 0.05, 0.5, 0.03])
    
    # Create sliders
    freq_slider = Slider(ax_freq, 'Frequency', 0.1, 5.0, 
                        valinit=freq_init, valfmt='%.1f Hz')
    amp_slider = Slider(ax_amp, 'Amplitude', 0.1, 3.0, 
                       valinit=amp_init, valfmt='%.1f')
    
    # Update function
    def update(val):
        freq = freq_slider.val
        amp = amp_slider.val
        line.set_ydata(amp * np.sin(2 * np.pi * freq * t))
        fig.canvas.draw_idle()
    
    # Connect sliders to update function
    freq_slider.on_changed(update)
    amp_slider.on_changed(update)
    
    # Reset button
    reset_ax = plt.axes([0.8, 0.1, 0.1, 0.05])
    reset_button = Button(reset_ax, 'Reset')
    
    def reset(event):
        freq_slider.reset()
        amp_slider.reset()
    
    reset_button.on_clicked(reset)
    
    plt.show()

def create_animated_plot():
    """
    Create animated plot showing wave propagation.
    
    Demonstrates:
    - Animation techniques
    - Real-time updates
    - Frame-based animation
    - Performance optimization
    """
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Animation parameters
    x = np.linspace(0, 4*np.pi, 200)
    
    # Create empty line objects
    line1, = ax.plot([], [], 'b-', linewidth=2, label='Wave 1')
    line2, = ax.plot([], [], 'r-', linewidth=2, label='Wave 2')
    line3, = ax.plot([], [], 'g--', linewidth=2, label='Interference')
    
    # Set up the plot
    ax.set_xlim(0, 4*np.pi)
    ax.set_ylim(-3, 3)
    ax.set_xlabel('Position')
    ax.set_ylabel('Amplitude')
    ax.set_title('Wave Interference Animation', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Animation function
    def animate(frame):
        # Time parameter
        t = frame * 0.1
        
        # Calculate waves
        wave1 = np.sin(x - 2*t)
        wave2 = 0.8 * np.sin(x + 1.5*t + np.pi/4)
        interference = wave1 + wave2
        
        # Update line data
        line1.set_data(x, wave1)
        line2.set_data(x, wave2)
        line3.set_data(x, interference)
        
        return line1, line2, line3
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=200, 
                                 interval=50, blit=True, repeat=True)
    
    plt.show()
    return anim  # Return to prevent garbage collection

def create_real_time_plot():
    """
    Create real-time data plotting simulation.
    
    Demonstrates:
    - Real-time data updates
    - Rolling window display
    - Performance optimization
    - Data streaming visualization
    """
    # Set up the figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Initialize data
    max_points = 100
    x_data = []
    y1_data = []
    y2_data = []
    
    # Create line objects
    line1, = ax1.plot([], [], 'b-', linewidth=2, label='Signal 1')
    line2, = ax2.plot([], [], 'r-', linewidth=2, label='Signal 2')
    
    # Set up axes
    ax1.set_xlim(0, max_points)
    ax1.set_ylim(-2, 2)
    ax1.set_title('Real-time Signal Monitoring', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Amplitude')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlim(0, max_points)
    ax2.set_ylim(-3, 3)
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Amplitude')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Animation function
    def update_data(frame):
        # Generate new data point
        x_data.append(frame)
        y1_data.append(np.sin(frame * 0.1) + 0.1 * np.random.randn())
        y2_data.append(np.cos(frame * 0.15) * np.exp(-frame * 0.01) + 
                      0.2 * np.random.randn())
        
        # Keep only recent data
        if len(x_data) > max_points:
            x_data.pop(0)
            y1_data.pop(0)
            y2_data.pop(0)
        
        # Update plots
        line1.set_data(x_data, y1_data)
        line2.set_data(x_data, y2_data)
        
        # Adjust x-axis if needed
        if len(x_data) >= max_points:
            ax1.set_xlim(x_data[0], x_data[-1])
            ax2.set_xlim(x_data[0], x_data[-1])
        
        return line1, line2
    
    # Create animation
    anim = animation.FuncAnimation(fig, update_data, frames=range(1000),
                                 interval=100, blit=True, repeat=False)
    
    plt.tight_layout()
    plt.show()
    return anim

if __name__ == "__main__":
    print("Running Interactive and Animated Plot Examples...")
    print("Note: Interactive features work best in interactive environments.")
    
    try:
        # Run examples (comment out interactive ones for non-interactive environments)
        create_interactive_plot()  # Comment out if running non-interactively
        anim1 = create_animated_plot()
        anim2 = create_real_time_plot()
        
        print("Interactive and animated examples completed!")
        print("Animations are running. Close plot windows to continue.")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()