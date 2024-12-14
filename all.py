import numpy as np
import random
import streamlit as st
import matplotlib.pyplot as plt

def generate_cities(num_cities):
    return np.random.rand(num_cities, 2)

def euclidean_distance(city1, city2):
    return np.sqrt(np.sum((city1 - city2) ** 2))

def total_distance(tour, cities):
    return sum(euclidean_distance(cities[tour[i]], cities[tour[(i + 1) % len(tour)]]) for i in range(len(tour)))

# Genetic Algorithm
def create_population(pop_size, num_cities):
    return [random.sample(range(num_cities), num_cities) for _ in range(pop_size)]

def crossover(parent1, parent2):
    child = [-1] * len(parent1)
    start, end = sorted(random.sample(range(len(parent1)), 2))
    child[start:end] = parent1[start:end]
    fill_pos = end
    for city in parent2:
        if city not in child:
            if fill_pos >= len(child):
                fill_pos = 0
            child[fill_pos] = city
            fill_pos += 1
    return child

def mutate(tour, mutation_rate):
    for i in range(len(tour)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(tour) - 1)
            tour[i], tour[j] = tour[j], tour[i]

def select_parents(population, cities):
    fitness = [1 / total_distance(tour, cities) for tour in population]
    return random.choices(population, weights=fitness, k=2)

def genetic_algorithm(cities, pop_size=100, generations=1000, mutation_rate=0.01):
    population = create_population(pop_size, len(cities))
    best_tour = min(population, key=lambda tour: total_distance(tour, cities))
    best_distance = total_distance(best_tour, cities)
    
    for _ in range(generations):
        new_population = []
        for _ in range(pop_size):
            parent1, parent2 = select_parents(population, cities)
            child = crossover(parent1, parent2)
            mutate(child, mutation_rate)
            new_population.append(child)
        population = new_population
        current_best_tour = min(population, key=lambda tour: total_distance(tour, cities))
        current_best_distance = total_distance(current_best_tour, cities)
        if current_best_distance < best_distance:
            best_tour, best_distance = current_best_tour, current_best_distance
    
    return best_tour, best_distance

# Particle Swarm Optimization
class Particle:
    def __init__(self, num_cities):
        self.position = random.sample(range(num_cities), num_cities)
        self.velocity = [random.randint(-1, 1) for _ in range(num_cities)]
        self.best_position = self.position[:]
        self.best_cost = float('inf')
    
    def update_velocity(self, global_best_position, inertia=0.5, cognitive=1.5, social=1.5):
        for i in range(len(self.velocity)):
            cognitive_velocity = cognitive * random.random() * (self.best_position[i] - self.position[i])
            social_velocity = social * random.random() * (global_best_position[i] - self.position[i])
            self.velocity[i] = inertia * self.velocity[i] + cognitive_velocity + social_velocity
    
    def update_position(self):
        for i in range(len(self.position)):
            if random.random() < self.velocity[i]:
                j = (i + random.randint(1, len(self.position) - 1)) % len(self.position)
                self.position[i], self.position[j] = self.position[j], self.position[i]

def particle_swarm_optimization(cities, num_particles=30, max_iter=1000, inertia=0.5, cognitive=1.5, social=1.5):
    particles = [Particle(len(cities)) for _ in range(num_particles)]
    global_best_position = None
    global_best_cost = float('inf')
    
    for _ in range(max_iter):
        for particle in particles:
            current_cost = total_distance(particle.position, cities)
            if current_cost < particle.best_cost:
                particle.best_position = particle.position[:]
                particle.best_cost = current_cost
            if current_cost < global_best_cost:
                global_best_position = particle.position[:]
                global_best_cost = current_cost
        
        for particle in particles:
            particle.update_velocity(global_best_position, inertia, cognitive, social)
            particle.update_position()
    
    return global_best_position, global_best_cost

# Simulated Annealing
def swap_cities(tour):
    new_tour = tour[:]
    i, j = random.sample(range(len(tour)), 2)
    new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
    return new_tour

def acceptance_probability(old_cost, new_cost, temperature):
    if new_cost < old_cost:
        return 1.0
    return np.exp((old_cost - new_cost) / temperature)

def simulated_annealing(cities, initial_temp=1000, cooling_rate=0.995, min_temp=1):
    current_tour = random.sample(range(len(cities)), len(cities))
    current_cost = total_distance(current_tour, cities)
    best_tour = current_tour[:]
    best_cost = current_cost
    temperature = initial_temp
    
    while temperature > min_temp:
        new_tour = swap_cities(current_tour)
        new_cost = total_distance(new_tour, cities)
        if acceptance_probability(current_cost, new_cost, temperature) > random.random():
            current_tour, current_cost = new_tour, new_cost
        if current_cost < best_cost:
            best_tour, best_cost = current_tour, current_cost
        temperature *= cooling_rate
    
    return best_tour, best_cost

def plot_tour(cities, tour):
    fig, ax = plt.subplots()
    for i in range(len(tour)):
        start_city = cities[tour[i]]
        end_city = cities[tour[(i + 1) % len(tour)]]
        ax.plot([start_city[0], end_city[0]], [start_city[1], end_city[1]], 'b-')
    ax.plot(cities[:, 0], cities[:, 1], 'ro')
    ax.set_title('Best Tour')
    return fig

# Streamlit UI
st.title('Traveling Problem Algorithms')

num_cities = st.number_input('Number of Cities', min_value=2, max_value=100, value=20, step=1)

# Genetic Algorithm parameters
st.subheader('Genetic Algorithm Parameters')
pop_size = st.slider('Population Size', min_value=10, max_value=500, value=100, step=10)
generations = st.slider('Generations', min_value=100, max_value=5000, value=1000, step=100)
mutation_rate = st.slider('Mutation Rate', min_value=0.001, max_value=0.1, value=0.01, step=0.001)

if st.button('Run Genetic Algorithm'):
    cities = generate_cities(num_cities)
    best_tour, best_distance = genetic_algorithm(cities, pop_size, generations, mutation_rate)
    st.write(f'Best Distance (Genetic Algorithm): {best_distance}')
    st.write(f'Best Tour (Genetic Algorithm): {best_tour}')
    fig = plot_tour(cities, best_tour)
    st.pyplot(fig)

# Particle Swarm Optimization parameters
st.subheader('Particle Swarm Optimization Parameters')
num_particles = st.slider('Number of Particles', min_value=10, max_value=100, value=30, step=5)
max_iter = st.slider('Maximum Iterations', min_value=100, max_value=5000, value=1000, step=100)
inertia = st.slider('Inertia', min_value=0.1, max_value=1.0, value=0.5, step=0.1)
cognitive = st.slider('Cognitive Component', min_value=0.5, max_value=3.0, value=1.5, step=0.1)
social = st.slider('Social Component', min_value=0.5, max_value=3.0, value=1.5, step=0.1)

if st.button('Run Particle Swarm Optimization'):
    cities = generate_cities(num_cities)
    best_tour, best_distance = particle_swarm_optimization(cities, num_particles, max_iter, inertia, cognitive, social)
    st.write(f'Best Distance (Particle Swarm Optimization): {best_distance}')
    st.write(f'Best Tour (Particle Swarm Optimization): {best_tour}')
    fig = plot_tour(cities, best_tour)
    st.pyplot(fig)

# Simulated Annealing parameters
st.subheader('Simulated Annealing Parameters')
initial_temp = st.slider('Initial Temperature', min_value=50, max_value=1000, value=100, step=5)
cooling_rate = st.slider('Cooling Rate', min_value=0.40, max_value=0.999, value=0.995, step=0.001)
min_temp = st.slider('Minimum Temperature', min_value=0.1, max_value=10.0, value=1.0, step=0.1)

if st.button('Run Simulated Annealing'):
    cities = generate_cities(num_cities)
    best_tour, best_distance = simulated_annealing(cities, initial_temp, cooling_rate, min_temp)
    st.write(f'Best Distance (Simulated Annealing): {best_distance}')
    st.write(f'Best Tour (Simulated Annealing): {best_tour}')
    fig = plot_tour(cities, best_tour)
    st.pyplot(fig)
