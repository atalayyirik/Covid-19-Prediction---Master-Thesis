# Covid-19 Prediction / Master Thesis

The codebase of prediction for Covid-19 cases throughout small village scenarios

- /Bottleneck_nn_data_1500-sim-4-day.txt file shows the simulation data which collected from simulation tool called "Vadere" including 1500 different random simulation case, each includes 4 consecutive day data of infected/susceptible cases.
Example data:
Day 1 Begin-End Rates:  0.01 0.99-0.03 0.97
Day 2 Begin-End Rates:  0.03 0.97-0.1 0.91
Day 3 Begin-End Rates:  0.1 0.91-0.21 0.79
Day 4 Begin-End Rates:  0.21 0.79-0.43 0.57
- /simulation.py file simulates the cases from console.
- /process.py file rounds and cleans data for model training.
- /sde is the main folder regarding the processing the data and training the models.
