import pygame
import neat
import random
import os

# Game constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
PIPE_WIDTH = 50
PIPE_SPACING = 200
PIPE_GAP = 100
BIRD_RADIUS = 15
BIRD_Y_CHANGE = 8
BIRD_COLOR = (255, 255, 0)

pygame.init()

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()

def create_pipe():
    pipe_height = random.randint(50, SCREEN_HEIGHT - PIPE_GAP - 50)
    upper_pipe = pygame.Rect(SCREEN_WIDTH, 0, PIPE_WIDTH, pipe_height)
    lower_pipe = pygame.Rect(SCREEN_WIDTH, pipe_height + PIPE_GAP, PIPE_WIDTH, SCREEN_HEIGHT)
    return {'upper': upper_pipe, 'lower': lower_pipe}

def play_game(net):
    bird_x = 50
    bird_y = SCREEN_HEIGHT // 2
    bird_y_change = 0

    pipes = [create_pipe()]

    score = 0
    running = True
    while running:
        screen.fill((0, 0, 0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        bird_y_change += 1
        bird_y += bird_y_change

        pygame.draw.circle(screen, BIRD_COLOR, (bird_x, bird_y), BIRD_RADIUS)

        for pipe in pipes:
            pipe['upper'].move_ip(-2, 0)
            pipe['lower'].move_ip(-2, 0)

            pygame.draw.rect(screen, (0, 255, 0), pipe['upper'])
            pygame.draw.rect(screen, (0, 255, 0), pipe['lower'])

        # Add new pipes when the right edge of the foremost pipe is 
# a certain distance from the right edge of the screen.
        SPACING_BETWEEN_PIPES = 300
        if pipes[0]['upper'].right < SCREEN_WIDTH - SPACING_BETWEEN_PIPES:
            pipes.append(create_pipe())

# Remove the foremost pipe when it has moved completely off the screen.
        if pipes[0]['upper'].right < 0:
            pipes.pop(0)

        next_pipe = pipes[0]
        gap_center = next_pipe['upper'].bottom + PIPE_GAP / 2
        inputs = (bird_y, gap_center - bird_y, next_pipe['upper'].left)

        output = net.activate(inputs)
        if output[0] > 0.5:
            bird_y_change = -BIRD_Y_CHANGE

        if bird_y - BIRD_RADIUS <= 0 or bird_y + BIRD_RADIUS >= SCREEN_HEIGHT:
            running = False

        for pipe in pipes:
            if pipe['upper'].colliderect(pygame.Rect(bird_x - BIRD_RADIUS, bird_y - BIRD_RADIUS, 2 * BIRD_RADIUS, 2 * BIRD_RADIUS)) or \
               pipe['lower'].colliderect(pygame.Rect(bird_x - BIRD_RADIUS, bird_y - BIRD_RADIUS, 2 * BIRD_RADIUS, 2 * BIRD_RADIUS)):
                running = False

        score += 1
        pygame.display.flip()
        clock.tick(60)

    return score

def eval_genomes(genomes, config):
    for _, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = play_game(net)

def run_neat(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())

    winner = population.run(eval_genomes, 50)

if __name__ == '__main__':
    local_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run_neat(config_path)
