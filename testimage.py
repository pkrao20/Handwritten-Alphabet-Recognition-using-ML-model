import pygame
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2

def predict_alphabet():
    # Load the Keras model
    model = keras.models.load_model('model1.h5')

    # Load the image
    image = cv2.imread('alphabet.png', 0)
    image = cv2.resize(image, (28, 28))  # Resize the image to the input shape of the model

    # Preprocess the imag
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize the pixel values to be between 0 and 1

    # Make predictions on the image
    predictions = model.predict(image)

    # Print the predicted class
    predicted_class = np.argmax(predictions)
    return predicted_class

def start_game():
    pygame.init()

    # Set up the font
    font = pygame.font.SysFont('Arial', 30)

    # Render the text

    # Set the size of the window
    screen = pygame.display.set_mode((600, 600))
    screen.fill((0,0,0))

    # Set the title of the window
    pygame.display.set_caption("Press S to Guess and C to clear")

    # Set the default color to white
    color = (255, 255, 255)

    # Set the thickness of the brush
    brush_size = 18.75

    # Set a boolean variable to indicate if the mouse button is pressed
    drawing = False

    # Start the main loop
    while True:
        # Check for events
        for event in pygame.event.get():
            # Quit the program if the window is closed
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            
            # Set drawing to True if the mouse button is pressed
            elif event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True
            
            # Set drawing to False if the mouse button is released
            elif event.type == pygame.MOUSEBUTTONUP:
                drawing = False
            
            # Draw a line if the mouse button is pressed and the mouse is moving
            elif event.type == pygame.MOUSEMOTION and drawing:
                pygame.draw.circle(screen, color, event.pos, brush_size)

        # Update the display
        pygame.display.update()

        # Save the image if the S key is pressed
        keys = pygame.key.get_pressed()
        if keys[pygame.K_s]:
            surface = pygame.display.get_surface()
            image = pygame.transform.scale(surface, (32, 32))
            global ind
            pygame.image.save(image, "alphabet.png")
            print("Image saved")
            ind = predict_alphabet()
            pygame.time.delay(500)
            mystr = "The   alphabet   is   " +  chr(ord('A')+ind)
            text = font.render(mystr, True, (220, 220, 220))
            text_rect = text.get_rect(center=(200, 20))
            screen.blit(text,text_rect)
            print(mystr)
            pygame.display.update()

        # clear screen
        if keys[pygame.K_c]: 
            screen.fill((0, 0, 0))
        
start_game()