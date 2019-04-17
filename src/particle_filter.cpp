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
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	default_random_engine gen;
	double std_x, std_y, std_theta; // Standard deviations for x, y, and theta

	// Set standard deviations for x, y, and theta
	std_x = std[0];
	std_y = std[1];
	std_theta = std[2];
	 
	num_particles = 1000;

	// This line creates a normal (Gaussian) distribution for x
	normal_distribution<double> dist_x(x, std_x);
	
	// Create normal distributions for y and theta
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);

	for (int i = 0; i < num_particles; ++i){
		double sample_x, sample_y, sample_theta;
		
		sample_x = dist_x(gen);
		sample_y = dist_y(gen);
		sample_theta = dist_theta(gen);	 
		
		Particle tempParticle;
		tempParticle.x = sample_x;
		tempParticle.y = sample_y;
		tempParticle.theta = sample_theta;
		tempParticle.weight = 1;
		particles.push_back(tempParticle);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	default_random_engine gen;
	double std_x, std_y, std_theta; // Standard deviations for x, y, and theta

	// Set standard deviations for x, y, and theta
	std_x = std_pos[0];
	std_y = std_pos[1];
	std_theta = std_pos[2];

	// This line creates a normal (Gaussian) distribution for x
	normal_distribution<double> dist_x(0, std_x);
	
	// Create normal distributions for y and theta
	normal_distribution<double> dist_y(0, std_y);
	normal_distribution<double> dist_theta(0, std_theta);

	for (auto &p : particles){

		// calculate new state
		if (fabs(yaw_rate) < 0.00001) {  
			p.x += velocity * delta_t * cos(p.theta);
			p.y += velocity * delta_t * sin(p.theta);
		} 
		else {
			p.x = p.x + (velocity/yaw_rate) * (sin(p.theta + (yaw_rate * delta_t)) - sin(p.theta));
			p.y = p.y + (velocity/yaw_rate) * (cos(p.theta) - cos(p.theta + (yaw_rate * delta_t)));
			p.theta = p.theta + (yaw_rate * delta_t);
		}

		p.x += dist_x(gen);
		p.y += dist_y(gen);
		p.theta += dist_theta(gen);	 
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for (auto &o : observations){
		int closestID = -1;
		double closestDist = std::numeric_limits<double>::max();

		for (auto &p : predicted){
			double distance = dist(o.x, o.y, p.x, p.y);
			if (distance < closestDist){
				closestID = p.id;
				closestDist = distance;
			}
		}
		o.id = closestID;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	weights.clear();

	for (auto &p : particles){
		vector<LandmarkObs> predicted;
		for (const auto &l : map_landmarks.landmark_list){
			if (dist(p.x, p.y, l.x_f, l.y_f) <= sensor_range){
				LandmarkObs tempLandmark;
				tempLandmark.id = l.id_i;
				tempLandmark.x = l.x_f;
				tempLandmark.y = l.y_f;
				predicted.push_back(tempLandmark);
			}
		}
		
		vector<LandmarkObs> obs_trans;
		for (const auto &o : observations){
			LandmarkObs tempTrans;
			tempTrans.id = o.id;
			tempTrans.x = p.x + (cos(p.theta) * o.x) - (sin(p.theta) * o.y);
			tempTrans.y = p.y + (sin(p.theta) * o.x) + (cos(p.theta) * o.y);
			obs_trans.push_back(tempTrans);
		}
		
		dataAssociation(predicted, obs_trans);

		p.weight = 1.0;
		for (const auto &ot : obs_trans){
			for (const auto &pred : predicted){
				if (pred.id == ot.id){
					double sig_x = std_landmark[0];
					double sig_y = std_landmark[1];
					
					// calculate normalization term
					double gauss_norm = (1/(2 * M_PI * sig_x * sig_y));

					// calculate exponent
					double exponent = (pow((ot.x - pred.x),2)/(2 * pow(sig_x,2))) + (pow((ot.y - pred.y),2)/(2 * pow(sig_y,2)));

					// calculate weight using normalization terms and exponent
					double obs_weight = gauss_norm * exp(-exponent);

					p.weight *= obs_weight;
				}
			}
		}
		weights.push_back(p.weight);
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	vector<Particle> resampled;
	
	// This line creates a uniform distribution for index
	default_random_engine gen;
	uniform_int_distribution<int> dist_index(0, num_particles-1);
	int index = dist_index(gen);
	double beta = 0;

	double max_weight = 0;
	for (const auto &w : weights){
		if (w > max_weight){
			max_weight = w;
		}
	}
	// auto max_weight = max_element(std::begin(weights), std::end(weights));
	uniform_real_distribution<double> dist_beta(0.0, 2.0*max_weight);
	for (int i = 0; i < num_particles; ++i){
		beta += dist_beta(gen);
		while (weights[index] < beta){
			beta -= weights[index];
			index++;
			if (index >= num_particles){
				index -= num_particles;
			}
		}
		resampled.push_back(particles[index]);
	}

	particles = resampled;
}

void ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

std::string ParticleFilter::getSenseCoord(Particle best, std::string coord) {
  std::string s{""};
  vector<double> v;
  if (coord == "X") { 
    v = best.sense_x;
  } else if(coord == "Y") {
    v = best.sense_y;
  } else {
    return "error";
  }
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}