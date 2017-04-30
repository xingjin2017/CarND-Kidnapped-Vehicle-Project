/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  default_random_engine gen;
  num_particles = 200;

  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
	
  for (int i = 0; i < num_particles; i++) {
    Particle particle;
    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1.0;
    particles.push_back(particle);
    weights.push_back(particle.weight);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
  default_random_engine gen;
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  for (int i = 0; i < num_particles; i++) {
    Particle& particle = particles[i];
    double ox = particle.x;
    double oy = particle.y;
    double ot = particle.theta;
    if (abs(yaw_rate) < 1e-3) {
      particle.x = particle.x + velocity*delta_t*cos(particle.theta) + dist_x(gen);
      particle.y = particle.y + velocity*delta_t*sin(particle.theta) + dist_y(gen);
      particle.theta = particle.theta + dist_theta(gen);
    } else {
      double x_f = particle.x + velocity/yaw_rate*(sin(particle.theta+yaw_rate*delta_t)
						   - sin(particle.theta));
      double y_f = particle.y + velocity/yaw_rate*(cos(particle.theta)
						   - cos(particle.theta+yaw_rate*delta_t));
      double theta_f = particle.theta + yaw_rate*delta_t;
      particle.x = x_f + dist_x(gen);
      particle.y = y_f + dist_y(gen);
      particle.theta = theta_f + dist_theta(gen);
    }
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
  for (int oi = 0; oi < observations.size(); oi++) {
    double min_distance = 50.0; // sensor range
    int min_pi = -1;
    for (int pi = 0; pi < predicted.size(); pi++) {
      double dx = predicted[pi].x - observations[oi].x;
      double dy = predicted[pi].y - observations[oi].y;
      double distance = sqrt(dx*dx+dy*dy);
      if (min_pi < 0 || distance < min_distance) {
	min_distance = distance;
	min_pi = pi;
      }
    }
    int oid = observations[oi].id;
    observations[oi].id = min_pi;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html
  std::vector<LandmarkObs> predicted;
  int id = 0;
  for (Map::single_landmark_s landmark : map_landmarks.landmark_list) {
    LandmarkObs landmarkObs;
    // Map landmark id starts with 1
    landmarkObs.id = id;
    landmarkObs.x = static_cast<double>(landmark.x_f);
    landmarkObs.y = static_cast<double>(landmark.y_f);
    predicted.push_back(landmarkObs);
    id++;
  }

  weights.clear();
  double weights_sum = 0.0;
  for (int i = 0; i < num_particles; i++) {
    double p_x = particles[i].x;
    double p_y = particles[i].y;
    double p_theta = particles[i].theta;
    std::vector<LandmarkObs> transformed_observations;
    for (int oi = 0; oi < observations.size(); oi++) {
      LandmarkObs obs = observations[oi];
      double x = observations[oi].x*cos(p_theta)
	- observations[oi].y*sin(p_theta) + p_x;
      double y = observations[oi].x*sin(p_theta)
	+ observations[oi].y*cos(p_theta) + p_y;
      obs.x = x;
      obs.y = y;
      transformed_observations.push_back(obs);
    }

    dataAssociation(predicted, transformed_observations);
    double weight = 1.0;
    for (int ti = 0; ti < transformed_observations.size(); ti++) {
      if (transformed_observations[ti].id >= 0) {
	double x = transformed_observations[ti].x;
	double y = transformed_observations[ti].y;
	double ux = predicted[transformed_observations[ti].id].x;
	double uy = predicted[transformed_observations[ti].id].y;
	double dxs = (x-ux)/std_landmark[0];
	double dys = (y-uy)/std_landmark[1];
	double f = exp(-(dxs*dxs + dys*dys)/2.0) / (2.0*M_PI*std_landmark[0]*std_landmark[1]);
	weight *= f;
      } else {
	// Not detected landmark, using max sensor range
	double dxs = sensor_range/std_landmark[0];
	double dys = sensor_range/std_landmark[1];
	double f = exp(-(dxs*dxs + dys*dys)/2.0) / (2.0*M_PI*std_landmark[0]*std_landmark[1]);
	weight *= f;
      }
    }

    particles[i].weight = weight;
    weights.push_back(weight);
    weights_sum += weight;
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<> d(weights.begin(), weights.end());
  std::vector<Particle> resampled;
  for (int i = 0; i < num_particles; i++) {
    int idx = d(gen);
    resampled.push_back(particles[idx]);
  }
  particles = resampled;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
