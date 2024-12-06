import os
import numpy as np
from pynput import mouse
from myUtils.RS import RealSenseCamera

def main():
    pass

if __name__ == "__main__":
    # Initialize pygame
    pygame.init()

    # Set up the screen
    # screen = pygame.display.set_mode((800, 600))  # Create a window of size 800x600
    # pygame.display.set_caption("Mouse Click Example")

    # Define colors
    # WHITE = (255, 255, 255)
    # RED = (255, 0, 0)

    # Initial position and radius of a circle
    # circle_pos = [400, 300]
    # circle_radius = 50

    # Main loop
    running = True
    while running:
        for event in pygame.event.get():
            # Check for quitting the window
            if event.type == pygame.QUIT:
                running = False
            
            # Check for mouse button clicks
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    print("Left mouse button clicked at:", event.pos)
                    circle_pos = list(event.pos)  # Move circle to the click position
                elif event.button == 3:  # Right mouse button
                    print("Right mouse button clicked at:", event.pos)
                    circle_radius += 10  # Increase circle radius
                elif event.button == 2:  # Middle mouse button
                    print("Middle mouse button clicked at:", event.pos)
                    circle_radius = max(10, circle_radius - 10)  # Decrease circle radius

        # Clear screen
        # screen.fill(WHITE)

        # Draw a circle
        # pygame.draw.circle(screen, RED, circle_pos, circle_radius)

        # Update the display
        # pygame.display.flip()

    # Quit pygame
    # pygame.quit()
