#include "common.h"
#include <cmath>
#include <vector>

using namespace std;

int gridSize;
vector<vector<vector<int>>> grid;

// Apply the force from neighbor to particle
void apply_force(particle_t& particle, particle_t& neighbor) {
    // Calculate Distance
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);

    // Very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

// Integrate the ODE
void move(particle_t& p, double size) {
    // Slightly simplified Velocity Verlet integration
    // Conserves energy better than explicit Euler method
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;

    // Bounce from walls
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }

    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
}

void compute_forces_grid(particle_t* parts, int num_parts) {
    for (int i = 0; i < gridSize; i++) {
        for (int j = 0; j < gridSize; j++) {
            for (int ind : grid[i][j]) {
                particle_t& p = parts[ind];
                //
                for (int di = -1; di <= 1; di++) {
                    for (int dj = -1; dj <= 1; dj++) {
                        //
                        int ni = i + di, nj = j + dj;
                        if (ni >= 0 && ni < gridSize && nj >= 0 && nj < gridSize) {
                            for (int neighborInd : grid[ni][nj]) {
                                //
                                if (ind != neighborInd) {
                                    particle_t& neighbor = parts[neighborInd];
                                    apply_force(p, neighbor);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void populate_grid(particle_t* parts, int num_parts) {
    for (int i = 0; i < gridSize; i++)
        for (int j = 0; j < gridSize; j++)
            grid[i][j].clear();

    for (int i = 0; i < num_parts; i++) {
        int gridX = int(parts[i].x / cutoff);
        int gridY = int(parts[i].y / cutoff);
        grid[gridX][gridY].push_back(i);
    }
}




void init_simulation(particle_t* parts, int num_parts, double size) {
	// You can use this space to initialize static, global data objects
    // that you may need. This function will be called once before the
    // algorithm begins. Do not do any particle simulation here

    gridSize = ceil(size / cutoff);
    grid.resize(gridSize, vector<vector<int>>(gridSize));
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // Compute Forces

    populate_grid(parts, num_parts);

    for (int i = 0; i < num_parts; ++i) {
        parts[i].ax = 0;
        parts[i].ay = 0;
    }

    compute_forces_grid(parts, num_parts);

    // Move Particles
    for (int i = 0; i < num_parts; ++i) {
        move(parts[i], size);
    }
}


