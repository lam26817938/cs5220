#include "common.h"
#include <omp.h>
#include <cmath>
#include <vector>
#include <algorithm>

// Put any static global variables here that you will use throughout the simulation.
using namespace std;

double gridCellSize;
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


void init_simulation(particle_t* parts, int num_parts, double size) {
    // You can use this space to initialize static, global data objects
    // that you may need. This function will be called once before the
    // algorithm begins. Do not do any particle simulation here

    gridCellSize=cutoff;
    gridSize = ceil(size / gridCellSize);
    grid.resize(gridSize, vector<vector<int>>(gridSize));

    for (int i = 0; i < num_parts; i++) {
        int gridX = int(parts[i].x / gridCellSize);
        int gridY = int(parts[i].y / gridCellSize);
        grid[gridX][gridY].push_back(i);
    }

}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // Compute Forces

    for (int x = 0; x < gridSize; ++x) {
        for (int y = 0; y < gridSize; ++y) {
            for (int ind : grid[x][y]) {
                particle_t& p = parts[ind];
                p.ax = 0;
                p.ay = 0;
                for (int di = -1; di <= 1; ++di) {
                    for (int dj = -1; dj <= 1; ++dj) {
                        int ni = x + di, nj = y + dj;
                        if (ni >= 0 && ni < gridSize && nj >= 0 && nj < gridSize) {
                            for (int neighborInd : grid[ni][nj]) {
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

    // Move Particles
    for (int i = 0; i < num_parts; ++i) {

        int x_old = int(parts[i].x / gridCellSize);
        int y_old = int(parts[i].y / gridCellSize);

        move(parts[i], size);

        int x_new = int(parts[i].x / gridCellSize);
        int y_new = int(parts[i].y / gridCellSize);


        if (x_old != x_new || y_old != y_new) {
            auto& oldCell = grid[x_old][y_old];
            oldCell.erase(remove(oldCell.begin(), oldCell.end(), i), oldCell.end());
            grid[x_new][y_new].push_back(i);

        }
    }

}
